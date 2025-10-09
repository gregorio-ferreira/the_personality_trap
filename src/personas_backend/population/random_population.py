import random

import pandas as pd  # type: ignore


def generate_questions_probabilities(ref_eval: pd.DataFrame) -> pd.DataFrame:
    q_prob = ref_eval.groupby(
        ["category", "key", "question", "question_number"], as_index=False
    ).agg(
        true_count=("answer", lambda x: (x == True).sum()),
        false_count=("answer", lambda x: (x == False).sum()),
        total=("answer", "count"),
    )

    # Compute probabilities and add them to the DataFrame
    q_prob["true_prob"] = q_prob["true_count"] / (q_prob["total"])
    q_prob["false_prob"] = q_prob["false_count"] / (q_prob["total"])

    return q_prob


def generate_random_questionnaire(q_prob: pd.DataFrame) -> pd.DataFrame:
    questionnaire = pd.DataFrame(
        columns=["category", "key", "question", "question_number", "answer"]
    )
    for _, row in q_prob.iterrows():
        probabilities = [row["true_prob"], row["false_prob"]]
        answer = random.choices([True, False], probabilities)[0]
        questionnaire = pd.concat(
            [
                questionnaire,
                pd.DataFrame(
                    {
                        "category": [row["category"]],
                        "key": [row["key"]],
                        "question": [row["question"]],
                        "question_number": [row["question_number"]],
                        "answer": [answer],
                    }
                ),
            ]
        )

    return questionnaire


def generate_random_questionnaires_from_distribution(
    ref_eval: pd.DataFrame, num_questionnaires: int = 826
) -> pd.DataFrame:
    # Generate multiple synthetic questionnaires
    random_questionnaire = pd.DataFrame(
        columns=["category", "key", "personality_id", "question_number", "answer"]
    )
    q_prob = generate_questions_probabilities(ref_eval)

    random.seed(42)

    for ii in range(num_questionnaires):
        _questionnaire = generate_random_questionnaire(q_prob)
        # Add a new column 'questionnaire_id' as the first column
        _questionnaire.insert(0, "personality_id", ii)

        random_questionnaire = pd.concat([random_questionnaire, _questionnaire])

    return random_questionnaire
