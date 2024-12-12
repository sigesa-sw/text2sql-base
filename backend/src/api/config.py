class Config:
    def __init__(self, debug=True, allow_llm_to_see_data=True):
        self.DEBUG = debug
        self.ALLOW_LLM_TO_SEE_DATA = allow_llm_to_see_data
        self.HOST = "0.0.0.0"
        self.PORT = 8084
        self.SWAGGER_URL = "/apidocs"