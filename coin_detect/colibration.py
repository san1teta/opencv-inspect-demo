class pixel_to_mm_calibrator:
    def __init__(self, pixels_per_mm):
        self.ppm = pixels_per_mm
    def pixels_to_mm(self, pixels):
        return pixels/self.ppm
