import math

import cv2 as cv
import numpy as np

from config import IsolatorConfig


class Obj:
    def __init__(self, pos, size, area, obj_id=None):
        self.pos = pos
        self.size = size
        self.area = area
        self.center = (int(pos[0]+size[0]/2), int(pos[1]+size[1]/2))

        self.id = obj_id

    def __sub__(self, other):
        return self.center[0] - other.center[0], self.center[1] - other.center[1]

    def __str__(self):
        return "Object of size {}x{} with coordinates: x = {}, y = {}".format(self.size[0], self.size[1],
                                                                              self.pos[0], self.pos[1])

    def distance(self, other):
        return (self.center[0] - other.center[0])**2 + (self.center[1] - other.center[1])**2

    def similarity(self, other):
        d = self.distance(other)
        a = (self.area - other.area)**2

        return a+d, d


class ObjectIsolator:

    @staticmethod
    def isolate(buffer):
        avg = ObjectIsolator._average(buffer)   # compute average

        object_positions = {}   # object positions are saved in this dict

        previous = None     # previous objects
        max_id = 0      # keep track of max id, can be extracted from object positions
        for one in buffer:
            if IsolatorConfig.debug:
                edit = cv.cvtColor(one, cv.COLOR_GRAY2RGB)  # only for debug purposes
            objects = ObjectIsolator._isolate_frame(one, avg)   # isolate objects
            for obj in objects:
                if not previous:    # assign new ID
                    max_id += 1  # beginning or empty previous iteration
                    obj.id = max_id  # + 1  # begins with ID 1

                else:   # find the most similar match

                    matches = ObjectIsolator._closest(obj, previous)

                    best = None
                    best_s = math.inf
                    for match in matches:
                        s, d = obj.similarity(match)
                        if d > IsolatorConfig.max_related_distance_square:  # break after arriving at max distance
                            break

                        if s < best_s:
                            best = match

                    if not best:
                        max_id += 1
                        obj.id = max_id
                    else:
                        obj.id = best.id
                        previous.remove(best)

                if obj.id in object_positions.keys():
                    object_positions[obj.id].append(obj.center)
                else:
                    object_positions[obj.id] = [obj.center]

                if IsolatorConfig.debug:
                    cv.rectangle(edit, obj.pos, (obj.pos[0]+obj.size[0], obj.pos[1]+obj.size[1]), (
                        255, obj.id**3 % 255, obj.id**5 % 255
                    ), 1)
                    cv.putText(edit, text="ID: "+str(obj.id), org=obj.center,
                               fontFace=cv.QT_FONT_NORMAL, fontScale=.5,
                               color=(255, 255, 255), thickness=1, lineType=cv.LINE_AA)

            previous = objects

            if IsolatorConfig.debug:
                cv.imshow('Frame', edit)
                if cv.waitKey(25) & 0xFF == ord('q'):
                    break

        return object_positions

    @staticmethod
    def _closest(origin, objects):
        objects.sort(key=lambda x: x.distance(origin))

        return objects

    @staticmethod
    def _isolate_frame(frame, avg):
        delta = cv.absdiff(avg, frame)  # compute absolute difference with average

        if IsolatorConfig.mode == 'threshold':
            _, threshold = cv.threshold(delta, IsolatorConfig.threshold_min,
                                        IsolatorConfig.threshold_max, cv.THRESH_BINARY)
        elif IsolatorConfig.mode == 'inRange':
            threshold = cv.inRange(delta, IsolatorConfig.threshold_min, IsolatorConfig.threshold_max)

        contours, hierarchy = cv.findContours(threshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        objects = []

        for contour in contours:
            area = cv.contourArea(contour)
            if IsolatorConfig.contour_min_area > area or area > IsolatorConfig.contour_max_area:
                continue

            (x, y, w, h) = cv.boundingRect(contour)

            objects.append(Obj(
                (x, y),
                (w, h),
                area
            ))

        return objects

    @staticmethod
    def _average(buffer):
        return np.mean(buffer, axis=0).astype(np.uint8)
