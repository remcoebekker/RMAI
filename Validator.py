from ultralytics import YOLO


class Validator:

    def validate_or_test(self, model_base_path: str, number_training_face_shots_tested: list, mode: str):
        # We loop through the different number of training face shots...
        for i in range(0, len(number_training_face_shots_tested)):
            # For each number of training face shots, there is a different trained model
            model_path = model_base_path + "face_shots_" + str(
                number_training_face_shots_tested[i]) + "/weights/best.pt"
            # And we classify all the identities in the test video...
            face_classifier = YOLO(model_path)
            name = mode + "_" + str(number_training_face_shots_tested[i])
            face_classifier.val(source=mode, name=name)