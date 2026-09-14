class Calibrator:
    def __init__(self, ppm: float) -> None:
        self.ppm = ppm
    def pixels_to_mm(self, pixels: int) -> float:
        return pixels/self.ppm
