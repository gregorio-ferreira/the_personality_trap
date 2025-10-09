import logging

import personas_backend.models_providers.models_utils as models_utils
from personas_backend.utils.config import ConfigManager

get_model_wrapper = models_utils.get_model_wrapper

# (path shim and imports already handled above)


class DummySession:
    def __init__(self, *_, **__):  # type: ignore[no-untyped-def]
        pass

    def get_credentials(self):  # type: ignore[no-untyped-def]
        class C:  # pragma: no cover - simple stub
            access_key = "X"
            secret_key = "Y"

        return C()

    def client(self, *_, **__):  # type: ignore[no-untyped-def]
        class DummyBedrock:  # pragma: no cover - stub
            def invoke_model(self, *_, **__):  # type: ignore[no-untyped-def]
                return {"body": {"modelOutputs": [{"content": [{"text": "{}"}]}]}}

        return DummyBedrock()


class DummyOpenAI:  # pragma: no cover - stub
    def __init__(self, *_, **__):  # type: ignore[no-untyped-def]
        pass

    class chat:  # type: ignore
        class completions:  # type: ignore
            @staticmethod
            def create(**_):  # type: ignore[no-untyped-def]
                class R:  # minimal response stub
                    def model_dump(self):  # type: ignore[no-untyped-def]
                        return {
                            "choices": [
                                {
                                    "finish_reason": "stop",
                                    "message": {"content": "{}"},
                                }
                            ],
                            "usage": {
                                "prompt_tokens": 1,
                                "completion_tokens": 1,
                                "total_tokens": 2,
                            },
                        }

                return R()


class DummyLogger(logging.Logger):
    def __init__(self):  # type: ignore[no-untyped-def]
        super().__init__("dummy")


def test_get_model_wrapper_openai(monkeypatch):  # type: ignore
    monkeypatch.setenv("PERSONAS_OPENAI__API_KEY", "TEST_KEY")
    monkeypatch.setenv("PERSONAS_OPENAI__ORG_ID", "ORG")
    # Patch OpenAI class
    monkeypatch.setenv("PYTHONWARNINGS", "ignore")
    import personas_backend.models_providers.openai_client as oc  # type: ignore  # noqa: E501

    monkeypatch.setattr(oc, "OpenAI", DummyOpenAI)
    logger = DummyLogger()
    cfg = ConfigManager()
    wrapper = get_model_wrapper("gpt4o", logger, config_manager=cfg)
    assert hasattr(wrapper, "model_id") and "gpt-4o" in wrapper.model_id


def test_get_model_wrapper_bedrock(monkeypatch):  # type: ignore
    monkeypatch.setenv("PERSONAS_BEDROCK__AWS_REGION", "eu-central-1")
    monkeypatch.setenv("PERSONAS_BEDROCK__AWS_CREDENTIALS", "dummy-profile")
    import personas_backend.models_providers.aws_bedrock_client as bc  # type: ignore  # noqa: E501

    class DummySessionFactory:  # pragma: no cover
        def __init__(self, *_, **__):  # type: ignore[no-untyped-def]
            pass

        def get_credentials(self):  # type: ignore[no-untyped-def]
            return object()

        def client(self, *_, **__):  # type: ignore[no-untyped-def]
            class DummyBedrock:  # pragma: no cover
                def invoke_model(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
                    return {"body": {"modelOutputs": [{"content": [{"text": "{}"}]}]}}

            return DummyBedrock()

    monkeypatch.setattr(bc.boto3.session, "Session", DummySessionFactory)  # type: ignore
    logger = DummyLogger()
    cfg = ConfigManager()
    wrapper = get_model_wrapper("llama323B", logger, config_manager=cfg)
    assert hasattr(wrapper, "model_id") and "llama3" in wrapper.model_id
