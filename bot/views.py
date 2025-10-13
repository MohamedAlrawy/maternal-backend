from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status, permissions
from django.conf import settings
from .models import QAPair, Section, ChatSession, ChatMessage
from .serializers import QAPairSerializer, SectionSerializer
from .utils import embed_text, find_best_match, cosine_similarity, ask_openai_chat
from django.shortcuts import get_object_or_404
from django.views.decorators.csrf import ensure_csrf_cookie
from rest_framework.decorators import api_view, permission_classes
import heapq

SIMILARITY_THRESHOLD = 0.78  # tune: 0.7-0.85 depending on embedding model

class ChatAPIView(APIView):
    permission_classes = [permissions.AllowAny]

    def post(self, request):
        user_message = request.data.get("message", "").strip()
        session_uuid = request.data.get("session_id")  # optional: passed from frontend
        
        if not user_message:
            return Response({"error": "message required"}, status=status.HTTP_400_BAD_REQUEST)

        # --- SESSION HANDLING ---
        print(1)
        if session_uuid:
            session_obj, _ = ChatSession.objects.get_or_create(session_id=session_uuid)
        else:
            # fallback to Django session
            if not request.session.session_key:
                request.session.create()
            session_uuid = request.session.session_key
            session_obj, _ = ChatSession.objects.get_or_create(session_id=session_uuid)
        
        
        # session setup
        session = request.session
        print(session)
        chat_history = session.get("chat_history", [])  # list of dicts [{"role":"user","text":...}, {"role":"assistant","text":...}, ...]

        # create embedding for message
        emb = embed_text(user_message)
        print(2)

        # build candidate list from QAPair embeddings
        qas = QAPair.objects.exclude(embedding__isnull=True)
        candidates = []
        for qa in qas:
            emb_qa = qa.get_embedding()
            if emb_qa:
                candidates.append((qa.id, emb_qa))

        # find best match
        best_id, best_score = find_best_match(emb, candidates) if candidates else (None, -1.0)
        print(3)
        # threshold check
        if best_id and best_score >= SIMILARITY_THRESHOLD:
            qa = QAPair.objects.get(id=best_id)
            answer = qa.answer
            source = "faq"
            matched_section = qa.section.header
            # include in session
            chat_history.append({"sender":"user","content":user_message})
            chat_history.append({"sender":"assistant","content":answer, "source": source, "matched_section": matched_section})
            session["chat_history"] = chat_history
            session.modified = True
            return Response({
                "reply": answer,
                "source": source,
                "matched_section": matched_section,
                "score": best_score
            })

        print(4)

        # else: ask OpenAI
        # prepare messages for OpenAI — include short system prompt, include session (limited last N turns), and include top-k relevant contexts
        system_prompt = (
            "You are a helpful medical assistant specialized in maternal and newborn health. "
            "Answer succinctly and clearly. If confident, cite the matched context provided. "
            "If uncertain or the question is beyond scope, say you don't know and recommend consulting a qualified clinician."
        )

        # find top-k relevant sections to include in prompt
        scored = []
        for qa in QAPair.objects.exclude(embedding__isnull=True):
            s = cosine_similarity(emb, qa.get_embedding())
            scored.append((s, qa))
        scored.sort(key=lambda x: x[0], reverse=True)
        top_k = [qa for _, qa in scored[:3] if _ > 0.0]

        context_snippets = []
        for qa in top_k:
            context_snippets.append(f"Header: {qa.section.header}\nContext: {qa.section.context}\nExample Q: {qa.question}\nExample A: {qa.answer}")

        # include recent session messages (last 6 turns)
        recent_turns = chat_history[-6:]
        messages = []
        for t in recent_turns:
            role = "user" if t.get("sender") == "user" else "assistant"
            messages.append({"sender": role, "content": t.get("content")})

        # append the user message at the end
        # build a prompt that includes the matched contexts
        user_payload = "User question: " + user_message
        if context_snippets:
            user_payload += "\n\nRelevant contexts:\n" + "\n\n---\n".join(context_snippets)

        messages.append({"role":"user", "content": user_payload})
        print(6)

        try:
            ai_reply = ask_openai_chat(system_prompt, messages)
        except Exception as e:
            return Response({"error": "OpenAI error", "details": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        try:
            ChatMessage.objects.create(session=session_obj, sender="user", content=user_message)
            ChatMessage.objects.create(session=session_obj, sender="bot", content=ai_reply, source="ai")
        except Exception as e:
            return Response({"error": "Database error", "details": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        print(7)
        # Save new dynamically generated QA? Optionally save to DB for later auditing
        # Save conversation in session
        chat_history.append({"sender":"user","content":user_message})
        chat_history.append({"sender":"assistant","content":ai_reply, "source":"ai"})
        session["chat_history"] = chat_history
        session.modified = True
        session.save()

        return Response({
            "answer": ai_reply,
            "message": ai_reply,
            "source": "ai",
            "matched_section": None,
            "score": best_score
        })

class ChatHistoryAPIView(APIView):
    permission_classes = [permissions.AllowAny]

    def get(self, request):
        session_uuid = request.query_params.get("session_id")
        if not session_uuid:
            return Response({"error": "session_uuid required"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            session_obj = ChatSession.objects.get(session_id=session_uuid)
        except ChatSession.DoesNotExist:
            return Response({"history": []})

        messages = ChatMessage.objects.filter(session=session_obj).values("sender", "content", "source", "matched_section", "created_at")
        return Response({"history": list(messages)})

    def delete(self, request):
        # clear history
        request.session["chat_history"] = []
        request.session.modified = True
        return Response({"status":"cleared"})
