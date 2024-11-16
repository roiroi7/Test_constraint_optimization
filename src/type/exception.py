class BaseAppException(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class UnexpectedTypeException(BaseAppException):
    msg: str

    def __init__(self, msg: str, *args: object) -> None:
        self.msg = msg
        super().__init__(*args)

    def __str__(self) -> str:
        return f"Unexpected type found: {self.msg}"
