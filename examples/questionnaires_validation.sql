SELECT
el.questionnaire,
    el.experiment_id,
    el.succeeded,
    el.llm_explanation IS NOT NULL as has_explanation,
    COUNT(eq.id) as answers,
    COUNT(erm.id) as has_metadata
FROM my_schema_v00.experiments_list el
LEFT JOIN my_schema_v00.eval_questionnaires eq ON eq.experiment_id = el.experiment_id
LEFT JOIN my_schema_v00.experiment_request_metadata erm ON erm.experiment_id = el.experiment_id
GROUP BY el.questionnaire, el.experiment_id, el.succeeded, el.llm_explanation
ORDER BY el.experiment_id DESC;


SELECT *
FROM my_schema_v00.experiment_request_metadata
ORDER BY id DESC;
