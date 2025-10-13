from django.core.management.base import BaseCommand
from bot.models import Section, QAPair
from docx import Document
from bot.utils import embed_text
import re

class Command(BaseCommand):
    help = "Ingest a Word (.docx) file with headers, contexts and Q/A lists."

    # def add_arguments(self, parser):
    #     parser.add_argument("docx_path", type=str)

    def handle(self, *args, **options):
        path = "bot/management/commands/chatbotdialogue.docx"
        doc = Document(path)

        # Simple parsing strategy:
        # - Treat paragraphs with style 'Heading' or length < 100 and bold as header.
        # - After a header, the next paragraph(s) until a Q/A list is context.
        # - Q/A lists are either paragraphs with "Doctor:" and "Bot:" or "Q:" and "A:" or " - Q: ... A: ..." patterns.
        # This is heuristic — adapt to your file structure.
        text_lines = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        i = 0
        created = 0
        while i < len(text_lines):
            line = text_lines[i]
            # detect header heuristically: all-caps or ends with ":" or short (<120)
            is_header = False
            if len(line) < 120 and (line.endswith(":") or line.isupper() or line.startswith("Header") or len(line.split()) <= 8):
                is_header = True
            # or if paragraph uses numbering like "1. Header"
            if re.match(r'^\d+\.\s+', line):
                is_header = True

            if is_header:
                header = line.rstrip(":").strip()
                i += 1
                # gather context until we hit Q/A style or another header
                context_lines = []
                qa_lines = []
                while i < len(text_lines):
                    nxt = text_lines[i]
                    # detect new header
                    nxt_is_header = (len(nxt) < 120 and (nxt.endswith(":") or nxt.isupper() or re.match(r'^\d+\.\s+', nxt)))
                    if nxt_is_header:
                        break
                    # detect QA marker
                    if re.search(r'^(Doctor[:\-]|Q:|Question:|Dr\.)', nxt, re.IGNORECASE) or re.search(r'^(Bot[:\-]|A:|Answer:)', nxt, re.IGNORECASE):
                        qa_lines.append(nxt)
                    else:
                        if qa_lines:
                            # if we have already started QA and this line is not QA, treat as QA continuation
                            qa_lines.append(nxt)
                        else:
                            context_lines.append(nxt)
                    i += 1

                context = "\n".join(context_lines).strip()
                # create section
                section = Section.objects.create(header=header, context=context)
                # parse QA lines into pairs. We'll look for lines that contain "Doctor" then following "Bot"
                j = 0
                buffer = []
                while j < len(qa_lines):
                    l = qa_lines[j]
                    # try patterns
                    match_q = re.match(r'^(Doctor[:\-\s]*|Q[:\-\s]*|Question[:\-\s]*)(.*)$', l, re.IGNORECASE)
                    if match_q:
                        q_text = match_q.group(2).strip()
                        # find next answer
                        ans = ""
                        k = j + 1
                        found_ans = False
                        while k < len(qa_lines):
                            m = qa_lines[k]
                            match_a = re.match(r'^(Bot[:\-\s]*|A[:\-\s]*|Answer[:\-\s]*)(.*)$', m, re.IGNORECASE)
                            if match_a:
                                ans = match_a.group(2).strip()
                                found_ans = True
                                k += 1
                                # also include subsequent lines that don't start with Doctor/Q
                                while k < len(qa_lines) and not re.match(r'^(Doctor[:\-\s]*|Q[:\-\s]*|Question[:\-\s]*)', qa_lines[k], re.IGNORECASE):
                                    ans += " " + qa_lines[k].strip()
                                    k += 1
                                break
                            else:
                                # sometimes the Q and A are on same line split by " - " or " / "
                                m2 = re.split(r'\s[-–—/]\s', l)
                                if len(m2) >= 2:
                                    q_text = m2[0].strip()
                                    ans = " - ".join(m2[1:]).strip()
                                    found_ans = True
                                    k = j + 1
                                    break
                                # else move on to next line
                            k += 1
                        if not found_ans:
                            # fallback: next line maybe the answer without label
                            if j + 1 < len(qa_lines):
                                ans = qa_lines[j+1]
                                k = j + 2
                            else:
                                ans = ""
                                k = j + 1
                        # create QAPair
                        qa = QAPair.objects.create(section=section, question=q_text, answer=ans)
                        # compute embedding for question+context to improve matching
                        emb_text = f"{section.header}\n{section.context}\nQuestion: {q_text}"
                        emb = embed_text(emb_text)
                        qa.set_embedding(emb)
                        qa.save()
                        created += 1
                        j = k
                    else:
                        # line might be "Q - A" in same line
                        parts = re.split(r'\s[-–—/]\s', l)
                        if len(parts) >= 2:
                            q_text = parts[0].strip()
                            ans = " - ".join(parts[1:]).strip()
                            qa = QAPair.objects.create(section=section, question=q_text, answer=ans)
                            emb_text = f"{section.header}\n{section.context}\nQuestion: {q_text}"
                            emb = embed_text(emb_text)
                            qa.set_embedding(emb)
                            qa.save()
                            created += 1
                        j += 1

                self.stdout.write(self.style.SUCCESS(f"Created section '{header}' and {section.qa_pairs.count()} QAs"))
            else:
                i += 1

        self.stdout.write(self.style.SUCCESS(f"Ingestion complete. Total QA created: {created}"))
