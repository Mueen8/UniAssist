"""
bot.py - UniAssistBot

UniAssist handles:

1. Normal ML-based university/class questions
2. Direct student-name questions
3. Partial student names
4. Roll-number searches
5. Student profile / CGPA / percentage / status questions
6. Class statistics

Student data is loaded separately from:
    students.json

The ML model continues to handle timetable, teachers, rooms,
university information, etc.
"""

import json
import re
from difflib import SequenceMatcher
from pathlib import Path

import joblib

from src.config import CONFIDENCE_THRESHOLD, FALLBACK_INTENT, MODEL_FILE
from src.data_loader import load_intents
from src.preprocessing import clean_text


class UniAssistBot:

    def __init__(self):

        # ---------------------------------------------------------
        # LOAD TRAINED ML MODEL
        # ---------------------------------------------------------

        if not MODEL_FILE.exists():
            raise FileNotFoundError(
                "Trained model not found. Please run: python train.py"
            )

        saved = joblib.load(MODEL_FILE)

        self.vectorizer = saved["vectorizer"]
        self.classifier = saved["classifier"]

        # Load normal intent responses
        _, _, self.responses = load_intents()

        # ---------------------------------------------------------
        # LOAD STUDENT DATA
        # ---------------------------------------------------------

        self.students = self._load_students()

    # =============================================================
    # STUDENT DATA
    # =============================================================

    def _load_students(self):
        """
        Load students.json.

        The code checks common locations.
        """

        possible_files = [
            Path("data/students.json"),
            Path("Students.json"),
            Path("students.json"),
        ]

        student_file = None

        for file_path in possible_files:
            if file_path.exists():
                student_file = file_path
                break

        if student_file is None:
            print(
                "WARNING: students.json not found. "
                "Student questions will not work."
            )
            return []

        try:
            with open(student_file, "r", encoding="utf-8") as file:
                data = json.load(file)

            students = data.get("students", [])

            if not isinstance(students, list):
                print(
                    "WARNING: 'students' must be a list "
                    "in students.json."
                )
                return []

            return students

        except json.JSONDecodeError as error:
            print(
                f"WARNING: students.json contains invalid JSON: {error}"
            )
            return []

        except Exception as error:
            print(
                f"WARNING: Could not load students.json: {error}"
            )
            return []

    # =============================================================
    # TEXT NORMALIZATION
    # =============================================================

    def _normalize(self, text):
        """
        Normalize text for student-name matching.
        """

        if not text:
            return ""

        text = str(text).lower()

        replacements = {
            "kon": "kaun",
            "koun": "kaun",
        }

        for old, new in replacements.items():
            text = re.sub(
                rf"\b{re.escape(old)}\b",
                new,
                text
            )

        # Remove punctuation
        text = re.sub(r"[^a-z0-9\s\-]", " ", text)

        # Normalize spaces
        text = re.sub(r"\s+", " ", text).strip()

        return text

    # =============================================================
    # STUDENT NAME TOKENS
    # =============================================================

    def _student_name_tokens(self, name):
        """
        Return individual tokens from student's name.

        Example:
        SYEDA UMME ABIHA
        ->
        ['syeda', 'umme', 'abiha']
        """

        normalized = self._normalize(name)

        return normalized.split()

    # =============================================================
    # FIND STUDENT CANDIDATES
    # =============================================================

    def _find_student_candidates(self, question):
        """
        Find students using:

        - Exact full name
        - Partial name
        - Individual name token
        - Roll number
        - Minor spelling mistakes
        """

        if not self.students:
            return []

        q = self._normalize(question)

        if not q:
            return []

        candidates = []

        # ---------------------------------------------------------
        # COMMON QUESTION WORDS
        # ---------------------------------------------------------

        stop_words = {
            "kaun",
            "kon",
            "koun",
            "hai",
            "hain",
            "he",
            "who",
            "is",
            "the",
            "about",
            "tell",
            "me",
            "please",
            "student",
            "ka",
            "ki",
            "ke",
            "kya",
            "batao",
            "bata",
            "profile",
            "info",
            "information",
            "cgpa",
            "gpa",
            "percentage",
            "percent",
            "marks",
            "roll",
            "number",
            "no",
            "status",
            "result",
            "class",
            "friend",
            "friends",
            "konse",
            "kis",
            "kiya",
        }

        words = q.split()

        useful_words = [
            word
            for word in words
            if word not in stop_words
            and len(word) >= 2
        ]

        # ---------------------------------------------------------
        # EXACT FULL NAME / ROLL NUMBER
        # ---------------------------------------------------------

        for student in self.students:

            name = self._normalize(
                student.get("name", "")
            )

            roll_no = self._normalize(
                student.get("roll_no", "")
            )

            if name and name in q:
                candidates.append((student, 100))

            elif roll_no and roll_no in q:
                candidates.append((student, 100))

        if candidates:
            return self._unique_candidates(candidates)

        # ---------------------------------------------------------
        # EXACT NAME TOKEN
        # ---------------------------------------------------------

        for student in self.students:

            name_tokens = self._student_name_tokens(
                student.get("name", "")
            )

            score = 0

            for word in useful_words:

                if word in name_tokens:
                    score = max(score, 90)

            if score > 0:
                candidates.append((student, score))

        if candidates:
            return self._unique_candidates(candidates)

        # ---------------------------------------------------------
        # FUZZY NAME MATCHING
        # ---------------------------------------------------------

        for student in self.students:

            name_tokens = self._student_name_tokens(
                student.get("name", "")
            )

            best_score = 0

            for word in useful_words:

                for token in name_tokens:

                    similarity = SequenceMatcher(
                        None,
                        word,
                        token
                    ).ratio()

                    if len(word) >= 4 and len(token) >= 4:

                        if similarity >= 0.82:

                            score = int(
                                similarity * 80
                            )

                            if score > best_score:
                                best_score = score

            if best_score > 0:
                candidates.append(
                    (student, best_score)
                )

        return self._unique_candidates(candidates)

    # =============================================================
    # UNIQUE STUDENT CANDIDATES
    # =============================================================

    def _unique_candidates(self, candidates):
        """
        Remove duplicate students and sort by score.
        """

        unique = {}

        for student, score in candidates:

            name = student.get(
                "name",
                ""
            ).upper()

            if (
                name not in unique
                or score > unique[name][1]
            ):
                unique[name] = (
                    student,
                    score
                )

        result = list(unique.values())

        result.sort(
            key=lambda item: item[1],
            reverse=True
        )

        return result

    # =============================================================
    # STUDENT QUESTION DETECTION
    # =============================================================

    def _looks_like_student_question(self, question):
        """
        Determine whether the question is about a student.
        """

        q = self._normalize(question)

        student_words = {
            "student",
            "kaun",
            "kon",
            "koun",
            "who",
            "profile",
            "cgpa",
            "gpa",
            "percentage",
            "percent",
            "marks",
            "roll",
            "rollno",
            "rollnumber",
            "status",
            "result",
            "classmate",
            "classmates",
            "friend",
            "friends",
            "banda",
            "larki",
            "girl",
            "boy",
        }

        words = set(q.split())

        if words.intersection(student_words):
            return True

        candidates = self._find_student_candidates(
            question
        )

        return len(candidates) > 0

    # =============================================================
    # STUDENT ANSWER
    # =============================================================

    def _student_answer(self, question):
        """
        Generate answer for individual student questions.

        failing_subjects is intentionally ignored.
        """

        candidates = self._find_student_candidates(
            question
        )

        if not candidates:
            return None

        # ---------------------------------------------------------
        # MULTIPLE MATCHES
        # ---------------------------------------------------------

        top_score = candidates[0][1]

        close_matches = [
            student
            for student, score in candidates
            if score >= top_score - 5
        ]

        if (
            len(close_matches) > 1
            and top_score < 100
        ):

            names = ", ".join(
                student.get(
                    "name",
                    "Unknown"
                )
                for student in close_matches[:5]
            )

            return (
                "Bhai, is naam se multiple students "
                "match ho rahe hain: "
                f"{names}. "
                "Thora sa aur naam bata dein, "
                "phir main exact student ki "
                "information de dunga."
            )

        student = candidates[0][0]

        name = student.get(
            "name",
            "Unknown Student"
        )

        roll_no = student.get(
            "roll_no",
            "Not available"
        )

        percentage = student.get(
            "percentage",
            "Not available"
        )

        cgpa = student.get(
            "cgpa",
            "Not available"
        )

        status = student.get(
            "status",
            "Not available"
        )

        profile = student.get(
            "profile",
            "Is student ki profile information "
            "available nahi hai."
        )

        q = self._normalize(question)

        # ---------------------------------------------------------
        # QUESTION TYPE
        # ---------------------------------------------------------

        asks_cgpa = (
            "cgpa" in q
            or "gpa" in q
        )

        asks_percentage = (
            "percentage" in q
            or "percent" in q
            or "marks" in q
        )

        asks_roll = (
            "roll" in q
            or "rollno" in q
            or "roll number" in q
            or "rollnumber" in q
        )

        asks_status = (
            "status" in q
            or "promoted" in q
            or "result" in q
        )

        asks_profile = (
            "kaun" in q
            or "kon" in q
            or "koun" in q
            or "who" in q
            or "profile" in q
            or "about" in q
            or "information" in q
            or "info" in q
            or "banda" in q
            or "larki" in q
            or "girl" in q
            or "boy" in q
        )

        # ---------------------------------------------------------
        # CGPA
        # ---------------------------------------------------------

        if asks_cgpa:

            return (
                f"{name} ka CGPA {cgpa} hai."
            )

        # ---------------------------------------------------------
        # PERCENTAGE
        # ---------------------------------------------------------

        if asks_percentage:

            return (
                f"{name} ki percentage "
                f"{percentage}% hai."
            )

        # ---------------------------------------------------------
        # ROLL NUMBER
        # ---------------------------------------------------------

        if asks_roll:

            return (
                f"{name} ka roll number "
                f"{roll_no} hai."
            )

        # ---------------------------------------------------------
        # STATUS
        # ---------------------------------------------------------

        if asks_status:

            return (
                f"{name} ka current status "
                f"{status} hai."
            )

        # ---------------------------------------------------------
        # PROFILE
        # ---------------------------------------------------------

        if asks_profile:

            return (
                f"{name} ({roll_no}) ka profile:\n"
                f"{profile}\n\n"
                f"Percentage: {percentage}%\n"
                f"CGPA: {cgpa}\n"
                f"Status: {status}"
            )

        # ---------------------------------------------------------
        # NAME ONLY
        # ---------------------------------------------------------

        return (
            f"{name} ke baare mein available "
            f"information:\n"
            f"{profile}\n\n"
            f"Roll No: {roll_no}\n"
            f"Percentage: {percentage}%\n"
            f"CGPA: {cgpa}\n"
            f"Status: {status}"
        )

    # =============================================================
    # CLASS INFORMATION & STATISTICS
    # =============================================================

    def _class_student_answer(self, question):
        """
        Handle class-level student information
        and academic statistics.
        """

        q = self._normalize(question)

        # ---------------------------------------------------------
        # TOTAL STUDENTS
        # ---------------------------------------------------------

        if (
            ("total" in q and "student" in q)
            or
            ("kitne" in q and "student" in q)
            or
            ("how many" in q and "student" in q)
        ):

            return (
                "BSCS 3rd E Evening mein total "
                f"{len(self.students)} students hain."
            )

        # ---------------------------------------------------------
        # AVERAGE CGPA
        # ---------------------------------------------------------

        if (
            "average cgpa" in q
            or "avg cgpa" in q
            or (
                "average" in q
                and "gpa" in q
            )
        ):

            cgpas = []

            for student in self.students:

                try:
                    cgpas.append(
                        float(
                            student.get("cgpa")
                        )
                    )

                except (TypeError, ValueError):
                    pass

            if cgpas:

                average_cgpa = (
                    sum(cgpas) / len(cgpas)
                )

                return (
                    f"Class ka average CGPA "
                    f"{average_cgpa:.2f} hai."
                )

        # ---------------------------------------------------------
        # AVERAGE PERCENTAGE
        # ---------------------------------------------------------

        if (
            "average percentage" in q
            or "avg percentage" in q
            or (
                "average" in q
                and "percent" in q
            )
        ):

            percentages = []

            for student in self.students:

                try:
                    percentages.append(
                        float(
                            student.get(
                                "percentage"
                            )
                        )
                    )

                except (TypeError, ValueError):
                    pass

            if percentages:

                average_percentage = (
                    sum(percentages)
                    / len(percentages)
                )

                return (
                    f"Class ki average percentage "
                    f"{average_percentage:.2f}% hai."
                )

        # ---------------------------------------------------------
        # HIGHEST CGPA
        # ---------------------------------------------------------

        if (
            "highest cgpa" in q
            or "maximum cgpa" in q
            or "max cgpa" in q
            or "highest gpa" in q
        ):

            valid_students = []

            for student in self.students:

                try:
                    cgpa = float(
                        student.get("cgpa")
                    )

                    valid_students.append(
                        (student, cgpa)
                    )

                except (TypeError, ValueError):
                    pass

            if valid_students:

                student, highest_cgpa = max(
                    valid_students,
                    key=lambda item: item[1]
                )

                return (
                    f"Highest CGPA "
                    f"{student.get('name')} ka hai, "
                    f"jo {highest_cgpa:.2f} hai."
                )

        # ---------------------------------------------------------
        # HIGHEST PERCENTAGE
        # ---------------------------------------------------------

        if (
            "highest percentage" in q
            or "maximum percentage" in q
            or "max percentage" in q
            or "highest percent" in q
        ):

            valid_students = []

            for student in self.students:

                try:
                    percentage = float(
                        student.get(
                            "percentage"
                        )
                    )

                    valid_students.append(
                        (student, percentage)
                    )

                except (TypeError, ValueError):
                    pass

            if valid_students:

                student, highest_percentage = max(
                    valid_students,
                    key=lambda item: item[1]
                )

                return (
                    f"Highest percentage "
                    f"{student.get('name')} ki hai, "
                    f"jo {highest_percentage:.2f}% hai."
                )

        # ---------------------------------------------------------
        # PROMOTED STUDENTS
        # ---------------------------------------------------------

        if (
            "promoted" in q
            and (
                "kitne" in q
                or "how many" in q
                or "number" in q
            )
        ):

            promoted = sum(
                1
                for student in self.students
                if str(
                    student.get(
                        "status",
                        ""
                    )
                ).strip().lower() == "promoted"
            )

            return (
                f"Class mein {promoted} students "
                f"Promoted hain."
            )

        return None

    # =============================================================
    # INTENT PREDICTION
    # =============================================================

    def predict_intent(self, question):
        """
        Return:
            (intent, confidence)
        """

        cleaned = clean_text(question)

        if not cleaned:
            return FALLBACK_INTENT, 0.0

        features = self.vectorizer.transform(
            [cleaned]
        )

        if features.nnz == 0:
            return FALLBACK_INTENT, 0.0

        probabilities = (
            self.classifier.predict_proba(
                features
            )[0]
        )

        best_index = probabilities.argmax()

        intent = str(
            self.classifier.classes_[best_index]
        )

        confidence = float(
            probabilities[best_index]
        )

        if confidence < CONFIDENCE_THRESHOLD:
            return (
                FALLBACK_INTENT,
                confidence
            )

        return intent, confidence

    # =============================================================
    # MAIN ANSWER FUNCTION
    # =============================================================

    def get_answer(self, question):
        """
        Main chatbot function.

        Priority:

        1. Class-level student questions
        2. Individual student questions
        3. Normal ML intent system
        4. Fallback
        """

        # ---------------------------------------------------------
        # 1. CLASS INFORMATION / STATISTICS
        # ---------------------------------------------------------

        class_answer = self._class_student_answer(
            question
        )

        if class_answer:

            return {
                "intent": "student_class_information",
                "confidence": 1.0,
                "answer": class_answer,
            }

        # ---------------------------------------------------------
        # 2. INDIVIDUAL STUDENT QUESTIONS
        # ---------------------------------------------------------

        if self._looks_like_student_question(
            question
        ):

            student_answer = self._student_answer(
                question
            )

            if student_answer:

                return {
                    "intent": "student_information",
                    "confidence": 1.0,
                    "answer": student_answer,
                }

        # ---------------------------------------------------------
        # 3. NORMAL ML INTENT SYSTEM
        # ---------------------------------------------------------

        intent, confidence = self.predict_intent(
            question
        )

        answer = self.responses.get(
            intent,
            self.responses[FALLBACK_INTENT]
        )

        return {
            "intent": intent,
            "confidence": confidence,
            "answer": answer,
        }