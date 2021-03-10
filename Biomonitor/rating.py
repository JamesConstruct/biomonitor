from config import RatingConfig


class Rating:

    @staticmethod
    def evaluate(predictions):
        n = len(predictions)
        if n == 0:
            return 4
        c = sum([x.argmax() for x in predictions])
        r = 1-c/n

        if n >= RatingConfig.population_threshold:
            if r >= RatingConfig.clean_to_all_ratio:
                return 1
            else:
                return 3
        else:
            if r >= RatingConfig.clean_to_all_ratio:
                return 2
            else:
                return 4
