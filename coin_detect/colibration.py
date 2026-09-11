class Calibrator:
    def __init__(self, ppm):
        self.ppm = ppm
    def pixels_to_mm(self, pixels):
        return pixels/self.ppm
