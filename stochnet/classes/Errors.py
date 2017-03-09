class ShapeError(Exception):
    """Exception raised for errors in the input shape.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message='Wrong shape!'):
        self.message = message
