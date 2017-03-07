class ShapeError(Exception):
    """Exception raised for errors in the input shape.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, message='Wrong shape!'):
        self.message = message
