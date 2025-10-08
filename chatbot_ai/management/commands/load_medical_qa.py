from django.core.management.base import BaseCommand
from chatbot_ai.models import QA


class Command(BaseCommand):
    help = 'Load sample medical Q&A data for maternal health chatbot'

    def handle(self, *args, **options):
        self.stdout.write('Loading medical Q&A data...')
        
        medical_qa_data = [
            {
                "question": "Doctor, I have a woman in labour who just arrived in the delivery suite. What should be my first steps?",
                "answer": "Begin with a calm and systematic assessment. Greet the patient, confirm her identity, and assess her general condition—consciousness, colour, breathing, and pain level. Then check vital signs: blood pressure, pulse, temperature, and oxygen saturation."
            },
            {
                "question": "What about the obstetric assessment?",
                "answer": "Start with a brief focused history—gravidity, parity, gestational age, onset of contractions, rupture of membranes, vaginal bleeding, and any risk factors such as hypertension, diabetes, or previous caesarean section."
            },
            {
                "question": "Should I perform an abdominal examination immediately?",
                "answer": "Yes. Palpate gently to determine the fundal height, fetal lie, presentation, and position. Assess the strength and frequency of contractions. Then auscultate the fetal heart rate for one full minute immediately after a contraction to detect decelerations."
            },
            {
                "question": "When should I perform a vaginal examination?",
                "answer": "Only after ensuring membranes are intact and there is no contraindication such as placenta previa. Use aseptic technique. Determine cervical dilation, effacement, station, membrane status, and presenting part. Document the findings carefully."
            },
            {
                "question": "Once I complete the assessment, what are the next steps?",
                "answer": "Establish an intravenous line with crystalloid fluids, obtain baseline investigations (CBC, blood group, and cross-match if risk factors exist). Record all findings on the partograph and commence continuous monitoring of both mother and fetus."
            },
            {
                "question": "How often should I monitor her progress?",
                "answer": "In the latent phase, monitor every four hours for progress and maternal vitals. Once in the active phase (cervix ≥4 cm with regular contractions), record progress every 30 minutes for fetal heart, hourly for maternal pulse, and every four hours for temperature and blood pressure. Vaginal examination is repeated every four hours or earlier if there are concerns."
            },
            {
                "question": "How do I know if the labour is progressing normally?",
                "answer": "On the partograph, the cervical dilation line should remain to the left of the alert line. Descent of the head should correspond with increasing dilation. Contractions should be regular, 3–4 every 10 minutes, each lasting 40–60 seconds, and the fetal heart rate should remain between 110–160 bpm."
            },
            {
                "question": "What if progress becomes slow?",
                "answer": "Re-evaluate possible causes—inefficient contractions, malposition, or full bladder. Encourage ambulation, hydration, and emptying of the bladder. If contractions are weak, you may start Oxytocin augmentation but only after ruling out cephalopelvic disproportion and obtaining senior approval. Always use an infusion pump and continuous fetal monitoring."
            },
            {
                "question": "What about pain management during labour?",
                "answer": "Pain relief is essential. Begin with non-pharmacologic methods—reassurance, continuous support, relaxation breathing, massage, and frequent position changes. If the patient desires medication, offer Entonox (nitrous oxide/oxygen mixture) during contractions, or opioid analgesia such as Pethidine 50–100 mg IM every 4 hours (avoid if delivery is imminent or in respiratory depression)."
            },
            {
                "question": "When should I consider epidural analgesia?",
                "answer": "Epidural is the most effective form of pain relief in labour. Offer it once the patient is in established labour and there are no contraindications such as coagulopathy, thrombocytopenia (<100,000), infection at the site, or patient refusal. It should be administered only by an anaesthetist under sterile conditions with continuous monitoring."
            },
            {
                "question": "How should I monitor after starting an epidural?",
                "answer": "Check blood pressure every 5 minutes for the first 15 minutes, then every 15 minutes thereafter. Monitor fetal heart continuously. Watch for hypotension—if it occurs, give 500 ml of warmed crystalloid and consider ephedrine 5–10 mg IV."
            },
            {
                "question": "What if labour becomes prolonged despite good contractions?",
                "answer": "Evaluate for obstruction or malposition. If the partograph crosses the alert line, inform the senior obstetrician. If it reaches the action line, plan intervention—either augmentation, assisted vaginal delivery, or caesarean section, depending on the case."
            },
            {
                "question": "How should I prepare for delivery when she reaches full dilation?",
                "answer": "Confirm full dilation, vertex at +2 or lower, intact reflexes, and good contractions. Prepare sterile delivery equipment, neonatal resuscitation area, and necessary medications for the third stage. Inform the midwife, paediatrician, and anaesthetist if high-risk."
            },
            {
                "question": "What should I remember for documentation?",
                "answer": "Record every observation—maternal vitals, fetal heart rate, contractions, drugs administered, and progress on the partograph. Accurate documentation is essential for both safety and audit purposes."
            },
            {
                "question": "When should I call for senior help?",
                "answer": "Always call for senior assistance if there are non-reassuring fetal heart patterns, slow progress, maternal complications, or any uncertainty in decision-making. Early escalation saves lives."
            },
            {
                "question": "Thank you, Doctor.",
                "answer": "You’re welcome. Consistent observation, communication, and timely intervention are the key elements of safe labour management."
            },
            {
                "question": "Doctor, my patient is bleeding heavily after delivery. What should I do first?",
                "answer": "This is a case of postpartum hemorrhage. Stay calm and act immediately. Call for help and activate the institutional PPH protocol. Assign clear roles to the team — one member monitors vital signs, another documents events, and a third prepares medications and equipment."
            },
            {
                "question": "Who should I inform right away?",
                "answer": "Inform the senior obstetrician, anesthetist, and blood bank. Alert the theatre team in case surgical intervention becomes necessary. Communication should be clear and concise: announce “Postpartum hemorrhage — active bleeding, estimated loss >500 ml.”"
            },
            {
                "question": "What initial actions should I take while help is coming?",
                "answer": "Begin uterine massage to stimulate contraction. Insert two large-bore (14–16G) IV cannulas. Draw blood for CBC, coagulation profile, fibrinogen, and cross-match at least six units of blood. Start warm crystalloids while awaiting blood. Ensure oxygen at 10 L/min via face mask."
            },
            {
                "question": "What is the first-line uterotonic I should give?",
                "answer": "Administer Oxytocin 5 IU IV slowly, followed by an infusion of 40 IU Oxytocin in 500 ml of normal saline at 125 ml/hour. Avoid giving a bolus faster than 5 IU to prevent hypotension."
            },
            {
                "question": "If bleeding continues after Oxytocin, what is next?",
                "answer": "Give Tranexamic Acid 1 gram IV over 10 minutes, ideally within 3 hours of delivery, to promote clot stability. It may be repeated once after 30 minutes if bleeding continues."
            },
            {
                "question": "Suppose the uterus remains atonic despite these measures — what should I do?",
                "answer": "Proceed stepwise. Administer Methylergometrine 0.2 mg IM if the patient is not hypertensive. If she has hypertension, skip to Carboprost 250 mcg IM every 15–20 minutes, up to a maximum of eight doses. Be cautious: avoid Carboprost in asthmatic patients due to risk of bronchospasm."
            },
            {
                "question": "And if bleeding persists even after those medications?",
                "answer": "Insert a Bakri balloon to tamponade the uterus if available. Inflate gradually with 300–500 ml of saline until bleeding slows. Monitor the amount in the drainage bag to detect ongoing loss. If bleeding continues despite the balloon, prepare for surgical intervention."
            },
            {
                "question": "When should I decide to go to the operating theatre?",
                "answer": "If there is no improvement within 15 minutes or if the patient becomes hemodynamically unstable, proceed immediately to the operating room. Delay increases mortality. In theatre, start with conservative surgical techniques."
            },
            {
                "question": "What surgical options should I consider?",
                "answer": "Begin with a B-Lynch compression suture if the uterus remains flabby. If bleeding continues, proceed to uterine artery ligation or internal iliac artery ligation if expertise is available. If all measures fail and the patient’s life is at risk, perform a subtotal or total hysterectomy. Always involve the senior obstetrician in this decision."
            },
            {
                "question": "What about transfusion and replacement?",
                "answer": "Activate the massive transfusion protocol if blood loss exceeds 1500 ml. The recommended ratio is 1:1:1 for packed red blood cells, plasma, and platelets. Transfuse cryoprecipitate if fibrinogen <2 g/L. Maintain calcium supplementation (10 ml of 10% calcium gluconate per 4 units of blood) to prevent citrate toxicity."
            },
            {
                "question": "What should the anesthetist do during this process?",
                "answer": "The anesthetist maintains airway, oxygenation, and hemodynamic stability. They administer fluids, blood products, and vasopressors if required. Continuous communication with anesthesia is essential to guide transfusion and resuscitation."
            },
            {
                "question": "What other causes should I consider apart from uterine atony?",
                "answer": "Always think of the '4 Ts': Tone — uterine atony (most common); Tissue — retained placental tissue or membranes; Trauma — cervical, vaginal, or perineal tears, or uterine rupture; Thrombin — coagulopathy or DIC. Inspect the perineum and cervix carefully, and ensure placenta is complete."
            },
            {
                "question": "How do I manage trauma-related bleeding?",
                "answer": "Identify the tear, visualize it under good light, and repair it in layers using absorbable sutures. If the uterus is ruptured, immediate laparotomy and surgical repair are required."
            },
            {
                "question": "What if retained tissue is suspected?",
                "answer": "If ultrasound or examination suggests retained placenta, perform manual removal under anesthesia. Give prophylactic antibiotics to prevent infection."
            },
            {
                "question": "How should I monitor the patient after bleeding is controlled?",
                "answer": "Continue close observation. Monitor vital signs every 15 minutes for the first hour, then hourly for six hours. Measure urine output hourly; maintain at least 0.5 ml/kg/hour. Repeat hemoglobin and coagulation studies. Administer broad-spectrum antibiotics, and continue uterotonics for 24 hours."
            },
            {
                "question": "What about documentation and debriefing?",
                "answer": "Document every event — estimated blood loss, medications, doses, procedures, and team members involved. Conduct a team debrief once the patient is stable. Discuss contributing factors and preventive strategies for future cases."
            },
            {
                "question": "Is there anything specific to counsel the patient about later?",
                "answer": "Yes. Explain the cause of the hemorrhage, what interventions were done, and discuss risks in future pregnancies. Advise delivery in a tertiary center with blood bank access for any subsequent pregnancy."
            },
            {
                "question": "Doctor, the patient is convulsing during labour, and she has a history of high blood pressure. What should I do first?",
                "answer": "This is an obstetric emergency — most likely eclampsia. Remain calm and act quickly. Call for help immediately. Inform the senior obstetrician, anesthetist, and neonatal team. Ensure patient safety by protecting the airway and preventing aspiration."
            },
            {
                "question": "How should I position her?",
                "answer": "Place her in the left lateral position to prevent aortocaval compression and improve venous return. Loosen tight clothing, remove sharp objects from around her, and never insert anything into her mouth during the seizure."
            },
            {
                "question": "What should I do once the seizure stops?",
                "answer": "Quickly assess airway, breathing, and circulation. Administer oxygen at 10 L/min by face mask. Check vital signs, and establish IV access with a large-bore cannula. Avoid giving large fluid boluses until you assess urine output and blood pressure."
            },
            {
                "question": "What medication should I start immediately?",
                "answer": "Begin Magnesium Sulphate therapy — it’s the drug of choice for seizure control and prevention. Give a loading dose of 4 grams IV over 15 minutes, followed by a maintenance infusion of 1 gram/hour IV for 24 hours after the last seizure or after delivery, whichever is later. If IV route isn’t available, use 10 grams IM (5 g in each buttock) with 1 ml of 2% lidocaine to reduce pain."
            },
            {
                "question": "What if she has another seizure despite magnesium therapy?",
                "answer": "Give a 2 gram IV bolus of magnesium sulphate over 5–10 minutes. If seizures persist, administer Diazepam 10 mg IV slowly, but only as a second-line option since it can depress respiration. Always ensure airway patency and monitor saturation."
            },
            {
                "question": "How should I monitor magnesium therapy?",
                "answer": "Monitor deep tendon reflexes, respiratory rate, and urine output hourly. If reflexes are lost, respiratory rate <12/min, or urine output <30 ml/hour, stop magnesium immediately — this indicates possible toxicity. Administer Calcium Gluconate 10 ml of 10% solution IV slowly over 10 minutes as an antidote."
            },
            {
                "question": "What are the blood pressure management targets?",
                "answer": "The goal is to maintain systolic BP between 140–150 mmHg and diastolic between 90–100 mmHg. For acute severe hypertension (≥160/110 mmHg), use any of the following options: Labetalol 20 mg IV over 2 minutes, repeat with 40 mg after 10 minutes, then 80 mg every 10 minutes (maximum 220 mg total). If ineffective, switch to Hydralazine 5–10 mg IV every 20–30 minutes. Oral Nifedipine 10 mg may be given if IV agents are unavailable."
            },
            {
                "question": "How should I manage fluids in eclampsia?",
                "answer": "Fluids must be restricted to avoid pulmonary edema. Maintain total fluids ≤80 ml/hour unless there’s active blood loss. Insert a Foley catheter and monitor hourly urine output — this helps detect renal impairment and guides fluid replacement."
            },
            {
                "question": "What investigations should I order?",
                "answer": "Send CBC, platelet count, liver and renal function tests, urine protein, coagulation profile, and electrolytes. If HELLP syndrome is suspected, monitor AST, ALT, LDH, and platelets closely."
            },
            {
                "question": "Once she stabilizes, what should be my next step?",
                "answer": "Plan for delivery — it’s the definitive treatment for eclampsia. If the patient is stable and in labour, allow vaginal delivery under close monitoring. If cervix is unfavourable or the condition worsens, proceed to caesarean section once maternal condition allows and with anesthesia support."
            },
            {
                "question": "Should I use epidural or general anesthesia for caesarean section?",
                "answer": "Epidural can be used if the patient is stable, reflexes are present, and coagulation is normal. If not, proceed with general anesthesia using rapid-sequence induction and airway precautions."
            },
            {
                "question": "How should I monitor her after delivery?",
                "answer": "Continue magnesium infusion for 24 hours after the last seizure or delivery. Monitor BP, reflexes, respiration, and urine output hourly. Maintain the patient in a high-dependency area. Continue antihypertensives as needed and avoid NSAIDs, which may worsen renal function."
            },
            {
                "question": "What complications should I watch for?",
                "answer": "Watch for pulmonary edema, renal failure, DIC, cerebral hemorrhage, and postpartum seizures. Any sudden deterioration in consciousness warrants immediate senior review and possibly a CT brain to exclude stroke."
            },
            {
                "question": "How should I counsel the patient afterward?",
                "answer": "Explain the diagnosis, treatment provided, and risk of recurrence in future pregnancies. Advise early booking and close antenatal follow-up in subsequent pregnancies. Encourage lifestyle modification and regular blood pressure monitoring."
            },
            {
                "question": "Thank you, Doctor.",
                "answer": "You’re welcome. Remember: in eclampsia, the priorities are airway, magnesium, blood pressure control, and timely delivery. Coordination between obstetrics, anesthesia, and critical care teams ensures maternal safety."
            },
            {
                "question": "Doctor, the baby’s head has delivered, but the shoulders are not coming with gentle traction. What should I do first?",
                "answer": "This is shoulder dystocia, an obstetric emergency caused by impaction of the fetal anterior shoulder behind the maternal symphysis pubis. Stay calm, call for immediate help, and announce 'Shoulder Dystocia' clearly to the team."
            },
            {
                "question": "Who should I call for help?",
                "answer": "Call the senior obstetrician, anesthetist, pediatrician, and experienced midwives. Assign clear roles: one to document, one to assist with maneuvers, and another to support the patient and maintain communication with the neonatal team."
            },
            {
                "question": "Should I ask the mother to continue pushing?",
                "answer": "No, stop maternal pushing immediately to prevent further impaction. Keep the head flexed and apply gentle traction in line with the fetal axis — do not pull excessively as this may cause brachial plexus injury."
            },
            {
                "question": "What are the first maneuvers I should attempt?",
                "answer": "Start with the McRoberts’ maneuver — hyperflex the mother’s hips tightly onto her abdomen to flatten the lumbar lordosis and rotate the symphysis pubis upward. This relieves the impaction in about 40% of cases. Simultaneously, apply suprapubic pressure in a downward and lateral direction behind the anterior shoulder."
            },
            {
                "question": "How long should I maintain McRoberts’ and suprapubic pressure?",
                "answer": "Maintain for 30–60 seconds. If the shoulder remains impacted, move on promptly — do not waste time repeating ineffective maneuvers. Every minute of delay increases the risk of fetal hypoxia and permanent injury."
            },
            {
                "question": "What is the next maneuver if this fails?",
                "answer": "Proceed to internal rotational maneuvers. Insert your hand along the sacral curve to reach the posterior shoulder. Attempt the Rubin II maneuver — push the posterior aspect of the anterior shoulder toward the fetal chest. If that fails, try the Woods screw maneuver by applying pressure to the anterior aspect of the posterior shoulder to rotate the fetus in a corkscrew fashion."
            },
            {
                "question": "Should I attempt both Rubin and Woods if one fails?",
                "answer": "Yes. Rubin II and Woods screw are complementary — apply Rubin II first, then rotate in the opposite direction using the Woods screw if needed. Use controlled, steady pressure to avoid humeral or clavicular fracture."
            },
            {
                "question": "What should I do if internal maneuvers do not work?",
                "answer": "Deliver the posterior arm next. Slide your hand along the baby’s back to locate the posterior elbow, flex it, and sweep the forearm and hand across the chest and face to deliver the posterior arm. This reduces the shoulder diameter and usually resolves the dystocia."
            },
            {
                "question": "Suppose the posterior arm also fails — what next?",
                "answer": "Reposition the mother to the all-fours (Gaskin) position if feasible — it can alter pelvic dimensions and help dislodge the shoulder. If this fails, prepare for last-resort measures: cleidotomy (intentional clavicle fracture), Zavanelli maneuver (cephalic replacement for cesarean), or symphysiotomy if the operator is experienced."
            },
            {
                "question": "What if I can’t remember the sequence under pressure?",
                "answer": "Use the HELPERR mnemonic: H – Call for Help, E – Evaluate for Episiotomy, L – Legs (McRoberts’), P – Pressure (suprapubic), E – Enter (Rubin or Woods), R – Remove posterior arm, R – Roll (all-fours position)."
            },
            {
                "question": "You mentioned episiotomy — should I do it early?",
                "answer": "Episiotomy does not directly relieve impaction but provides more room for internal maneuvers. It’s useful if access is limited, especially in first-time mothers or when the perineum is tight."
            },
            {
                "question": "What should I avoid during this emergency?",
                "answer": "Never apply fundal pressure — it worsens impaction and increases fetal injury risk. Avoid excessive traction on the head; keep all movements controlled and in line with the fetal axis."
            },
            {
                "question": "How should I document the event afterward?",
                "answer": "Document the exact time of head delivery and total head-to-body interval. Record each maneuver attempted and its duration, the personnel involved, estimated blood loss, and neonatal condition at birth. Documentation must be objective, detailed, and time-stamped."
            },
            {
                "question": "What immediate care does the newborn need after delivery?",
                "answer": "Hand the baby immediately to the neonatal team for assessment. They should check Apgar scores, assess arm movements for brachial plexus injury, and check for fractures. Begin resuscitation if the baby is not breathing or cyanosed."
            },
            {
                "question": "How should I manage the mother afterward?",
                "answer": "Inspect the birth canal for trauma or bleeding, repair any tears, and monitor vitals closely. Debrief the mother and her family once she’s stable, explaining what occurred, possible neonatal complications, and the follow-up plan."
            },
            {
                "question": "What should I advise about future deliveries?",
                "answer": "Document the shoulder dystocia clearly in her record. In future pregnancies, assess fetal size and risk factors early. If estimated fetal weight ≥4.5 kg in diabetics or ≥5 kg in non-diabetics, consider elective cesarean section."
            },
            {
                "question": "Thank you, Doctor.",
                "answer": "You’re welcome. Shoulder dystocia requires composure, teamwork, and prompt action. Remember — early recognition, clear communication, and a structured approach save lives and prevent injury."
            },
            {
                "question": "Doctor, I’ve just examined the patient after her membranes ruptured, and I can feel the umbilical cord in the vagina. What should I do immediately?",
                "answer": "This is cord prolapse — an obstetric emergency. Stay calm and act instantly. Keep your examining hand in place to relieve pressure from the presenting part, and call for immediate help."
            },
            {
                "question": "Who should I call first?",
                "answer": "Call the senior obstetrician, anesthetist, and neonatal team at once. Ask the midwife to press the emergency bell and prepare for immediate delivery. Inform the operating theatre that a Category 1 caesarean section is required."
            },
            {
                "question": "What should I do while waiting for help to arrive?",
                "answer": "Maintain manual elevation of the presenting part using two fingers or the whole hand through the vagina to prevent cord compression. Do not remove your hand until a definitive measure such as bladder filling or caesarean section is ready."
            },
            {
                "question": "How should I position the patient?",
                "answer": "Place her in a steep Trendelenburg or knee-chest position with hips elevated and chest down. This allows gravity to reduce pressure on the cord. If that’s not feasible, use the left lateral position with a pillow under the hip."
            },
            {
                "question": "Can I handle the cord directly?",
                "answer": "Avoid handling the cord unnecessarily because manipulation causes vasospasm. If the cord is protruding outside the vagina, cover it with warm sterile saline-soaked gauze to prevent drying and maintain warmth."
            },
            {
                "question": "Should I fill the bladder?",
                "answer": "Yes. If help is delayed or transfer to theatre will take more than a few minutes, instill 500–700 ml of warm normal saline into the bladder through a Foley catheter and clamp it. This pushes the fetal head upward and relieves cord compression."
            },
            {
                "question": "What about tocolytics — should I use them?",
                "answer": "You may give a short-acting tocolytic to reduce contractions while preparing for delivery. Terbutaline 0.25 mg subcutaneously or Nitroglycerin 50–100 µg IV bolus can be used, but these should never delay transfer to theatre."
            },
            {
                "question": "How should I monitor the fetus during this time?",
                "answer": "Use continuous electronic fetal monitoring if possible. Expect variable or prolonged decelerations. Reassess fetal heart rate after each maneuver to evaluate improvement."
            },
            {
                "question": "What is the mode of delivery?",
                "answer": "Emergency caesarean section is the preferred method for most viable fetuses. Aim for a decision-to-delivery interval under 30 minutes — sooner if fetal heart tones are non-reassuring."
            },
            {
                "question": "What if the cervix is fully dilated and the fetal head is low?",
                "answer": "If the cervix is fully dilated and the head is at +2 station or lower, consider instrumental vaginal delivery (vacuum or forceps) only if an experienced obstetrician is present and delivery can be achieved safely within minutes. Otherwise, proceed directly to caesarean section."
            },
            {
                "question": "What should the anesthetist do while we prepare?",
                "answer": "The anesthetist should secure the airway, ensure maternal oxygenation, and prepare for rapid-sequence induction. Continue manual elevation until the uterine incision is made and delivery is imminent."
            },
            {
                "question": "How do I manage after the baby is delivered?",
                "answer": "Hand the neonate immediately to the pediatric team for resuscitation. Expect possible hypoxia — the baby may require positive-pressure ventilation or advanced neonatal care. After delivery, give the mother uterotonics as per routine third-stage management and monitor for postpartum hemorrhage."
            },
            {
                "question": "What should I document about this emergency?",
                "answer": "Record the time of diagnosis, fetal heart rate pattern, maternal positions used, drugs given, and time of delivery. Note all personnel present and interventions performed. Documentation must be detailed — this case has medicolegal importance."
            },
            {
                "question": "How should I counsel the patient afterward?",
                "answer": "Explain that cord prolapse is unpredictable and not due to any action she took. Emphasize that the rapid response was lifesaving. Advise that future deliveries should occur in a facility with surgical capability and continuous fetal monitoring, especially if risk factors recur."
            },
            {
                "question": "Thank you, Doctor.",
                "answer": "You’re welcome. Remember, time is critical in cord prolapse — immediate manual elevation, proper positioning, and prompt decision for delivery make the difference between a healthy baby and a tragic outcome."
            },
            {
                "question": "Doctor, my patient was in active labour after a previous caesarean section, and suddenly she’s complaining of severe abdominal pain. The contractions have stopped, and I can’t feel the fetal head easily. What should I do?",
                "answer": "These are classic signs of a uterine rupture — an obstetric catastrophe requiring immediate action. Stop any oxytocin infusion if running, call for senior obstetrician, anesthetist, and theatre team, and prepare for emergency laparotomy."
            },
            {
                "question": "What other signs should alert me to uterine rupture?",
                "answer": "Watch for sudden cessation of contractions, abnormal fetal heart tracing (bradycardia or decelerations), loss of fetal station, abdominal tenderness, and vaginal bleeding. Sometimes you may palpate fetal parts easily through the abdomen or note maternal tachycardia and hypotension indicating concealed hemorrhage."
            },
            {
                "question": "Are there specific risk factors I should keep in mind?",
                "answer": "Yes. Major risk factors include previous classical or low-vertical caesarean scar, induction or augmentation with prostaglandins or oxytocin, obstructed labour, grand multiparity, uterine trauma or previous myomectomy, and overdistension of the uterus (such as multiple pregnancy or polyhydramnios)."
            },
            {
                "question": "What should I do immediately while the team is coming?",
                "answer": "Ensure oxygen at 10 L/min via face mask. Insert two large-bore IV lines, send blood for CBC, cross-match at least 6 units, coagulation profile, and start warm crystalloids. Keep the patient nil per mouth and insert a Foley catheter to monitor urine output. If there is evidence of shock, begin resuscitation with fluids and request urgent blood products."
            },
            {
                "question": "Should I perform a vaginal examination?",
                "answer": "No. Once rupture is suspected, avoid further vaginal examination — it can worsen bleeding and delay life-saving surgery. Focus on stabilizing the patient and arranging immediate transfer to the operating theatre."
            },
            {
                "question": "What should I communicate to the team and family?",
                "answer": "Briefly but clearly explain the gravity of the situation: 'There is a strong suspicion of uterine rupture, which is life-threatening for both mother and baby. We need to proceed urgently to surgery.' Obtain verbal consent if possible, but never delay for written consent when the patient’s life is at risk."
            },
            {
                "question": "What should be prepared in theatre?",
                "answer": "Prepare for exploratory laparotomy under general anesthesia. Ensure blood, suction, diathermy, and appropriate surgical instruments are available. Have senior obstetric and anesthesia support present. The pediatric team should be ready for neonatal resuscitation."
            },
            {
                "question": "What are the key surgical findings to expect?",
                "answer": "You may find a complete tear in the lower uterine segment extending into the broad ligament, cervix, or vagina. Sometimes, the fetus and placenta may be partly or completely expelled into the abdominal cavity. Hemoperitoneum is often significant."
            },
            {
                "question": "How do I manage during surgery?",
                "answer": "After delivering the baby and placenta, control active bleeding by clamping uterine and internal iliac arteries if necessary. Assess the extent of the tear. If the edges are clean and repairable, perform two-layer uterine repair with absorbable sutures. However, if the tear is extensive, ragged, or the patient is hemodynamically unstable, proceed to subtotal or total hysterectomy. Always prioritize saving the mother’s life."
            },
            {
                "question": "Should I attempt to conserve the uterus in all cases?",
                "answer": "Only if bleeding can be controlled and the patient is stable. In grand multiparas, extensive rupture, or failed previous repair, hysterectomy is the safer option. The decision should be taken by the senior obstetrician based on intraoperative findings."
            },
            {
                "question": "How do we manage blood loss intraoperatively?",
                "answer": "Activate the massive transfusion protocol if necessary. Replace blood and components in a 1:1:1 ratio (packed cells, plasma, platelets). Correct coagulopathy promptly with cryoprecipitate if fibrinogen <2 g/L. Maintain urine output at least 0.5 ml/kg/hr, and monitor for DIC."
            },
            {
                "question": "What about the baby?",
                "answer": "Perinatal mortality is high in complete rupture. Hand the neonate immediately to the pediatric team for resuscitation. Document Apgar scores, cord pH, and any complications."
            },
            {
                "question": "How should I manage the mother postoperatively?",
                "answer": "Keep her in a high-dependency unit. Continue close monitoring of vital signs, urine output, bleeding, and hemoglobin. Give broad-spectrum antibiotics for at least 5 days, uterotonics if the uterus was preserved, and thromboprophylaxis when stable."
            },
            {
                "question": "What complications should I watch for?",
                "answer": "Postoperative complications include hemorrhagic shock, sepsis, coagulopathy, wound infection, bladder or ureteric injury, and psychological trauma. Watch also for renal failure due to hypoperfusion."
            },
            {
                "question": "How do I counsel the patient later?",
                "answer": "Once she recovers, explain clearly that a uterine rupture occurred, the reasons for surgical management, and implications for future pregnancies. Advise that any future conception should occur only in a tertiary hospital, with elective caesarean section at 36–37 weeks before labour begins. Strongly advise against trial of labour in the future."
            },
            {
                "question": "Thank you, Doctor.",
                "answer": "You’re welcome. Uterine rupture demands immediate recognition, rapid resuscitation, and surgical expertise. The key to prevention is careful selection for VBAC, judicious use of uterotonics, and constant vigilance during labour."
            },
            {
                "question": "Doctor, a term patient suddenly became acutely short of breath and hypotensive during labour. She is anxious, cyanosed, and her oxygen saturation dropped to 80% despite oxygen. What should I suspect?",
                "answer": "Suspect amniotic fluid embolism (AFE)—an abrupt, anaphylactoid reaction to amniotic fluid or fetal material entering the maternal circulation, classically presenting with sudden hypoxia, hypotension, and coagulopathy/DIC. Treat this as a Category 1 obstetric emergency.",
            },
            {
                "question": "Doctor, a 36-week pregnant patient has suddenly collapsed in the labour room. She is unresponsive. What are my immediate priorities?",
                "answer": "Treat this as maternal cardiac arrest until proven otherwise. Shout for help, pull the emergency bell, and activate the arrest team. Begin high-quality CPR immediately with pregnancy modifications. Ensure someone establishes time-keeping and documentation from the first minute.",
            },
            {
                "question": "Doctor, 30 minutes have passed since delivery and the placenta has not delivered. What should I do?",
                "answer": "This meets the definition of retained placenta. Remain calm and act systematically. Call for senior assistance, ensure IV access, and start a PPH preparedness approach in case bleeding escalates. Confirm vitals, palpate the uterus for tone, and assess bleeding."
            },
            {
                "question": "What immediate bedside measures should I take for retained placenta?",
                "answer": "Ensure the bladder is empty—insert or drain a Foley catheter if needed. Apply controlled cord traction only if the uterus is firm and contracted and counter-traction is applied above the pubic symphysis. Do not pull on the cord if the uterus is atonic or there is resistance, as this risks uterine inversion."
            },
            {
                "question": "Should I give any medication at this stage?",
                "answer": "If the uterus is atonic, give a uterotonic to improve tone: Oxytocin 5 IU IV slowly then 40 IU in 500 ml infusion. If there is ongoing bleeding, give Tranexamic acid 1 g IV over 10 minutes, may repeat once after 30 minutes. If tone improves and the placenta separates, controlled cord traction may succeed."
            },
            {
                "question": "What if the placenta still does not separate?",
                "answer": "Consider placenta adherens (incomplete separation) versus placenta accreta spectrum (abnormally adherent). If there is no separation and bleeding is minimal, prepare for manual removal of placenta (MROP) in the operating theatre under adequate analgesia/anesthesia."
            },
            {
                "question": "How do I proceed with manual removal of the placenta safely?",
                "answer": "Move to the theatre. Inform anesthesia, prepare broad-spectrum antibiotics (e.g., Cefazolin 1–2 g IV; in penicillin allergy use Clindamycin 900 mg IV), position the patient, and ensure good lighting. With a sterile gloved hand, follow the umbilical cord to the placental bed, find the cleavage plane, and gently peel the placenta off the uterine wall using the edge of your hand, supporting the fundus externally with the other hand. Remove membranes completely."
            },
            {
                "question": "What if placenta accreta is suspected during manual removal?",
                "answer": "Stop attempts at forcible separation. Persistent adherence without a clear plane, heavy bleeding with minimal separation, or previous scar should raise suspicion. Call the senior obstetrician and escalate to conservative hemostatic measures (balloon tamponade) or surgical options. If bleeding is life-threatening and conservative measures fail, proceed to hysterectomy. Arrange interventional radiology if available and activate massive transfusion protocol."
            },
            {
                "question": "How do I manage bleeding during and after removal of the placenta?",
                "answer": "Continue uterotonics (oxytocin infusion; consider Methylergometrine 0.2 mg IM unless hypertensive; Carboprost 250 mcg IM q15–20 min up to 8 doses—avoid in asthma; Misoprostol 800–1000 mcg PR as adjunct). If atony persists, insert a Bakri balloon (inflate 300–500 ml) and monitor drainage. Maintain warmed fluids and blood products as indicated by labs (aim fibrinogen ≥2 g/L)."
            },
            {
                "question": "What should I document and communicate after managing retained placenta?",
                "answer": "Record the time since birth, measures attempted, medications (dose and time), decision for MROP, intraoperative findings (degree of adherence), blood loss, products transfused, and postoperative plan. Communicate with the patient and family once stable, explaining the cause, procedures performed, and implications for future pregnancies."
            },
            {
                "question": "What postoperative care is required after manual removal of placenta?",
                "answer": "Observe in a high-acuity area initially. Monitor vitals, urine output, uterine tone, and vaginal bleeding. Continue antibiotics for 24 hours (longer if contamination or extensive manipulation). Provide thromboprophylaxis when bleeding risk allows. Arrange follow-up for anemia treatment and counselling."
            },
            {
                "question": "During third stage, the patient suddenly has severe pain, the uterus disappears from the abdomen, and there is heavy bleeding with shock. What should I suspect?",
                "answer": "Suspect an acute uterine inversion—a rare but catastrophic emergency. It presents with hemorrhage, vagal shock, and absence of a palpable fundus. Announce the emergency, call the senior obstetrician and anesthetist, and prepare the theatre."
            },
            {
                "question": "What are the first critical actions in uterine inversion?",
                "answer": "Do not remove the placenta if still attached, and stop all uterotonics (oxytocin, methylergometrine, carboprost). Begin aggressive resuscitation—two large-bore IV cannulas, bloods for CBC, coagulation, fibrinogen, cross-match, start warmed crystalloids, and give oxygen. Consider Tranexamic acid 1 g IV if significant bleeding."
            },
            {
                "question": "How do I correct uterine inversion?",
                "answer": "Attempt the Johnson maneuver immediately at the bedside: grasp the inverted fundus with the palm and fingers and push it upward through the cervix along the long axis toward the umbilicus, using steady pressure. Maintain continuous pressure until the uterus passes the constriction ring and returns to its anatomical position."
            },
            {
                "question": "What if the cervix is constricted and I cannot replace the fundus?",
                "answer": "Ask anesthesia to provide tocolysis or deep anesthesia to relax the uterus and cervix: Nitroglycerin 50–100 µg IV boluses (can repeat) or Terbutaline 0.25 mg SC. Alternatively, halogenated inhalational agents under GA can facilitate relaxation. Once relaxation is achieved, retry the Johnson maneuver immediately."
            },
            {
                "question": "What other techniques can be used if manual replacement fails?",
                "answer": "Move to theatre for the Hydrostatic (O’Sullivan) method: instill warm saline into the vagina using a sealed system with the patient in Trendelenburg; hydrostatic pressure may reverse the inversion. If unsuccessful and the cervix is constricted, surgical correction is required—Huntington (traction on round ligaments) or Haultain (posterior cervical ring incision) procedures under laparotomy."
            },
            {
                "question": "When should uterotonics be given during uterine inversion?",
                "answer": "Only after the uterus has been successfully repositioned. Then give Oxytocin infusion immediately to maintain tone and prevent re-inversion, and consider Methylergometrine (unless hypertensive) or Carboprost (avoid in asthma) as needed. Maintain bimanual compression temporarily."
            },
            {
                "question": "How do I manage hemorrhage during uterine inversion?",
                "answer": "Treat as major PPH: activate massive transfusion protocol, transfuse components in balanced ratios guided by labs, correct fibrinogen to ≥2 g/L, and replace calcium. If atony persists after repositioning and uterotonics, insert a Bakri balloon. Maintain active warming and correct acidosis."
            },
            {
                "question": "Do I remove the placenta before or after repositioning the uterus?",
                "answer": "After. If the placenta is still attached, do not attempt removal until the uterus has been repositioned and is contracting—removal beforehand worsens bleeding and impedes replacement."
            },
            {
                "question": "What anaesthetic considerations are important in uterine inversion?",
                "answer": "Anticipate hemodynamic instability and rapid blood loss. Anesthesia should secure the airway, support circulation with fluids/vasopressors, and provide the uterine relaxation needed for reduction. Communicate every step and prepare for rapid transition from relaxation to uterotonic maintenance post-reduction."
            },
            {
                "question": "What should I document after managing uterine inversion?",
                "answer": "Record the time of inversion, vital signs, resuscitative measures, drugs given with doses, reduction maneuvers used (Johnson/hydrostatic/surgical), timing of successful reposition, bleeding estimates, transfusion volumes, and postoperative plan. Note whether the placenta was attached and the time of its removal."
            },
            {
                "question": "How do I counsel the patient after recovery from uterine inversion?",
                "answer": "Explain clearly that an acute uterine inversion occurred, the steps taken to reverse it, and any transfusions or surgery performed. Discuss the risk of recurrence in future deliveries (low but present), emphasize active management of the third stage, avoidance of excessive cord traction, and the need for delivery in a hospital with experienced staff. Provide psychological support and schedule follow-up."
            },
            {
                "question": "Doctor, I’m concerned my patient might be septic. She’s febrile at 38.9°C, tachycardic 125/min, and feels unwell. What should I do first?",
                "answer": "Treat this as suspected maternal sepsis. Escalate immediately—call the senior obstetrician, anesthetist, and inform the neonatal team if the fetus is viable. Begin the Sepsis 1-hour bundle: obtain cultures, start broad-spectrum antibiotics, measure serum lactate, begin early fluids if hypotensive or lactate ≥2 mmol/L, and arrange continuous monitoring."
            },
            {
                "question": "How do I recognize sepsis formally in obstetrics?",
                "answer": "Use physiologic red flags and obstetric early warning scores (MEOWS/NEWS2): temperature ≥38°C or <36°C, heart rate >110/min, respiratory rate >24/min, systolic BP <90 mmHg or MAP <65, altered mental status, SpO₂ <94%, oliguria <0.5 ml/kg/hr, rigors, or severe abdominal/pelvic pain. In pregnancy/postpartum, a normal white count does not exclude sepsis—clinical judgment is paramount."
            },
            {
                "question": "What is my immediate sequence at the bedside?",
                "answer": "1) Airway & Breathing: give oxygen to maintain SpO₂ 94–98%. 2) Circulation: establish two large-bore IVs; attach cardiac monitor and NIBP. 3) Labs: draw two sets of blood cultures (if it won't delay antibiotics), CBC, CRP, electrolytes, creatinine, LFTs, coagulation, ABG/venous gas with lactate, and type & screen; obtain urine and site cultures. 4) Fetal assessment: continuous CTG if undelivered and viable. 5) Antibiotics: start within 60 minutes. 6) Fluids: if hypotensive or lactate ≥2, give 30 ml/kg balanced crystalloid, reassessing frequently."
            },
            {
                "question": "Which empiric antibiotics should I start if the source is unclear?",
                "answer": "Follow local policy. Reasonable empiric options: Piperacillin–tazobactam 4.5 g IV q6h, OR Ceftriaxone 2 g IV daily + Metronidazole 500 mg IV q8h. Add Vancomycin if MRSA risk or severe soft-tissue/line infection. Adjust with microbiology and culture results."
            },
            {
                "question": "If I suspect chorioamnionitis in labour, what is preferred?",
                "answer": "Ampicillin 2 g IV q6h + Gentamicin 5 mg/kg IV daily. If Caesarean or anaerobic coverage is needed, add Clindamycin 900 mg IV q8h or Metronidazole. For severe penicillin allergy consider Clindamycin + Gentamicin."
            },
            {
                "question": "And postpartum endometritis?",
                "answer": "Clindamycin 900 mg IV q8h + Gentamicin 5 mg/kg IV daily is standard. Continue until afebrile and clinically improved for 24–48 hours, then step down to oral therapy as appropriate."
            },
            {
                "question": "What about UTI/pyelonephritis in pregnancy or postpartum?",
                "answer": "Ceftriaxone 1–2 g IV daily (or Cefotaxime 1 g IV q8h). In β-lactam allergy consider Aztreonam. Once afebrile 24–48 hours and clinically improved, step down to an oral, culture-guided agent to complete 10–14 days for pyelonephritis."
            },
            {
                "question": "We gave the first antibiotic dose. She’s hypotensive at 85/50 with lactate 3.2 mmol/L. How should I resuscitate?",
                "answer": "Proceed with early goal-directed resuscitation. Give up to 30 ml/kg balanced crystalloid total, titrating in 250–500 ml boluses with frequent reassessment (mental status, capillary refill, BP/MAP, lung exam, urine output). Use smaller boluses and ultrasound guidance if preeclampsia or cardiac disease is a concern."
            },
            {
                "question": "She remains hypotensive despite 2 liters of fluid.",
                "answer": "Start a vasopressor—Norepinephrine is first-line to target MAP ≥65 mmHg. You may start peripherally via a large bore IV while arranging central access. If refractory, consider adding Vasopressin (0.03 units/min) or Epinephrine per ICU guidance. Insert arterial line when feasible."
            },
            {
                "question": "What are the targets for organ perfusion?",
                "answer": "MAP ≥65 mmHg, urine output ≥0.5 ml/kg/hr, falling lactate trend, warm peripheries, and improving mental status. Repeat lactate in 2–4 hours to assess response."
            },
            {
                "question": "Beyond antibiotics and hemodynamics, what about source control?",
                "answer": "Source control is essential: remove retained products (evacuation), drain infected CS wound/hematoma, IR or surgical drainage for pelvic abscess, remove infected lines and culture tips, and arrange urology for obstructive pyelonephritis. Do not delay definitive source control once resuscitation is underway."
            },
            {
                "question": "How should I manage fluids in preeclampsia or suspected pulmonary oedema risk?",
                "answer": "Use conservative boluses (e.g., 250 ml) with frequent reassessment. Use bedside lung ultrasound for B-lines/IVC assessment if available. Escalate to high-flow nasal oxygen or NIV and involve anesthesia/ICU early if respiratory status worsens."
            },
            {
                "question": "What ongoing monitoring is required?",
                "answer": "Admit to HDU/ICU if shock or organ dysfunction. Monitor vitals every 15 minutes initially, continuous SpO₂/ECG, hourly urine output via Foley, and lab trends (CBC, U&E, LFTs, coagulation) every 6–12 hours or per severity. Maintain glucose 140–180 mg/dL, correct electrolytes, and start VTE prophylaxis once bleeding risk allows."
            },
            {
                "question": "Should I continue fetal monitoring if she’s undelivered and viable?",
                "answer": "Yes—continuous CTG while maternal resuscitation proceeds. If maternal instability is refractory or fetal status is non-reassuring despite optimization, discuss expedited delivery with senior team; timing depends on gestation, maternal response, and infection source."
            },
            {
                "question": "How long should antibiotic therapy continue?",
                "answer": "Typical durations: 7–10 days for bacteremia or deep pelvic infection; 3–5 days for uncomplicated lower UTI once clinically improved; 10–14 days for pyelonephritis or complicated endometritis. De-escalate according to cultures and clinical response."
            },
            {
                "question": "Are there situations where tranexamic acid is indicated in septic patients?",
                "answer": "TXA is not routinely indicated for sepsis-related coagulopathy. Prioritize source control, antibiotics, haemodynamic stabilization, and component therapy if DIC with bleeding. Use TXA only for concurrent obstetric haemorrhage when clinically appropriate."
            },
            {
                "question": "What maternal complications should I anticipate and prevent?",
                "answer": "ARDS, acute kidney injury, DIC, cardiomyopathy, stroke, and post-sepsis syndrome. Prevent with meticulous oxygenation, hemodynamic support, early source control, lung-protective ventilation if intubated, and early mobilization during recovery."
            },
            {
                "question": "How should I communicate with the family?",
                "answer": "Provide clear, honest updates: suspected sepsis diagnosis, actions taken (cultures, antibiotics within the hour, fluids, vasopressors), need for ICU, and fetal monitoring or delivery planning. Appoint a single senior spokesperson and document discussions with times."
            },
            {
                "question": "What exactly must I document for audit and safety?",
                "answer": "Record time of sepsis recognition, MEOWS/NEWS2 triggers, cultures taken, time of first antibiotic and agent, lactate values, fluid volumes, vasopressor start/time/doses, source control procedures, fetal status, team members, response to therapy, and any complications."
            },
            {
                "question": "Doctor, 30 minutes have passed since delivery and the placenta has not delivered. What should I do?",
                "answer": "This meets the definition of retained placenta. Remain calm and act systematically. Call for senior assistance, ensure IV access, prepare for possible PPH, confirm maternal vitals, palpate uterine tone, and assess bleeding."
            },
            {
                "question": "What immediate bedside measures should I take?",
                "answer": "Ensure the bladder is empty—insert or drain a Foley if needed. Apply controlled cord traction only if the uterus is firm and contracted and counter-traction is applied above the pubic symphysis. Do not pull on the cord if the uterus is atonic or resistance is felt (risk of inversion)."
            },
            {
                "question": "Should I give any medication at this stage?",
                "answer": "If the uterus is atonic give a uterotonic (Oxytocin 5 IU IV slowly followed by 40 IU in 500 ml infusion). If ongoing bleeding, give Tranexamic Acid 1 g IV over 10 minutes (may repeat once after 30 minutes). If tone improves and the placenta separates, controlled cord traction may succeed."
            },
            {
                "question": "What if the placenta still does not separate?",
                "answer": "Consider placenta adherens vs placenta accreta spectrum. If no separation and bleeding is minimal, prepare for manual removal of placenta (MROP) in theatre under adequate analgesia/anesthesia."
            },
            {
                "question": "How do I proceed with manual removal safely?",
                "answer": "Move to theatre, inform anesthesia, give prophylactic antibiotics (e.g., Cefazolin 1–2 g IV; Clindamycin if penicillin allergic), ensure sterile technique and good lighting. With a sterile gloved hand follow the cord to the placental bed, find the cleavage plane, gently separate the placenta while supporting the fundus externally, and remove membranes completely."
            },
            {
                "question": "What if placenta accreta is suspected during MROP?",
                "answer": "Stop forcible separation. Persistent adherence, heavy bleeding with minimal separation, or prior uterine scar raises suspicion. Call senior obstetrician, consider conservative measures (balloon tamponade), arrange IR (balloon occlusion/embolization) if available, and activate MTP. If life-threatening haemorrhage persists, proceed to hysterectomy."
            },
            {
                "question": "How do I manage bleeding during and after removal?",
                "answer": "Continue uterotonics (oxytocin infusion first; consider Methylergometrine unless hypertensive; Carboprost unless asthmatic; Misoprostol adjunct). If atony persists insert a Bakri balloon (300–500 ml) and monitor drainage. Maintain warmed fluids and blood products guided by labs; aim fibrinogen ≥2 g/L."
            },
            {
                "question": "During third stage, I felt a sudden severe pain reported by the patient, the uterus disappeared from the abdomen, and there is heavy bleeding with shock. What should I suspect?",
                "answer": "Suspect an acute uterine inversion—an emergency characterized by haemorrhage, vagal shock, and absence of a palpable fundus. Announce the emergency, call senior obstetrician and anaesthetist, and prepare for theatre and resuscitation."
            },
            {
                "question": "How do I correct the inversion?",
                "answer": "Attempt the Johnson maneuver immediately at the bedside: grasp the inverted fundus with the palm and fingers and push it upward through the cervix toward the umbilicus, maintaining steady pressure until the uterus passes the constriction ring and returns to its anatomical position."
            },
            {
                "question": "What if the cervix is constricted and I cannot replace the fundus?",
                "answer": "Ask anesthesia for uterine relaxation (Nitroglycerin 50–100 µg IV boluses or Terbutaline 0.25 mg SC) or deep inhalational anaesthesia. Retry the Johnson maneuver promptly once the uterus is relaxed."
            },
            {
                "question": "Should I consider other techniques if Johnson fails?",
                "answer": "If manual replacement fails, move to theatre for the hydrostatic (O’Sullivan) method using warm saline instilled into the vagina with a sealed system. If unsuccessful and the constriction ring remains, proceed to surgical correction (Huntington or Haultain) under laparotomy."
            },
            {
                "question": "When do I give uterotonics again?",
                "answer": "Only after successful repositioning. Immediately start an Oxytocin infusion to maintain tone and prevent re-inversion; consider Methylergometrine (unless hypertensive) or Carboprost (avoid if asthmatic) as needed. Provide temporary bimanual compression as required."
            },
            {
                "question": "Do I remove the placenta before or after reposition?",
                "answer": "After. Do not attempt placental removal before repositioning—the placenta should only be removed once the uterus is back in position and contracting."
            },
            {
                "question": "What should I document?",
                "answer": "Record time of inversion, vital signs, resuscitative measures, drugs and doses, reduction maneuvers used (Johnson/hydrostatic/surgical), time of successful reposition, estimated blood loss, transfusion volumes, placenta status and removal time, and postoperative plan."
            }
        ]
        created_count = 0
        updated_count = 0
        
        for qa_data in medical_qa_data:
            qa, created = QA.objects.get_or_create(
                question=qa_data['question'],
                defaults={
                    'answer': qa_data['answer'],
                    'category': qa_data.get('category', 'general'),
                    'keywords': qa_data.get('keywords', '')
                }
            )
            
            if created:
                created_count += 1
                self.stdout.write(
                    self.style.SUCCESS(f'✓ Created: {qa.question[:60]}...')
                )
            else:
                updated_count += 1
                self.stdout.write(
                    self.style.WARNING(f'- Already exists: {qa.question[:60]}...')
                )
        
        self.stdout.write(
            self.style.SUCCESS(f'\n✓ Successfully processed {created_count + updated_count} Q&A pairs!')
        )
        self.stdout.write(
            self.style.SUCCESS(f'  - Created: {created_count}')
        )
        self.stdout.write(
            self.style.WARNING(f'  - Already existed: {updated_count}')
        )

