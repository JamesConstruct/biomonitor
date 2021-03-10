
class CameraConfig:
    index = 0
    resolution = None  # (1920, 1080)
    simulation_source = ''
    simulation_begin = 0
    gray = True


class IsolatorConfig:
    threshold_min = 15
    threshold_max = 255
    mode = "inRange"

    contour_min_area = 20
    contour_max_area = 5000

    max_related_distance_square = 25**2

    debug = False


class ClassifierConfig:
    trajectory_min_length = 20
    model_path = 'weights/model'
    learning_rate = 0.00001


class MainConfig:
    fps = 30
    buffer_size = 5*fps


class RatingConfig:
    population_threshold = 5
    clean_to_all_ratio = 0.7

