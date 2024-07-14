import cv2
import pandas as pd
from ultralytics import YOLO


class FaceClassifier:
    FONT = cv2.FONT_HERSHEY_SIMPLEX

    def __init__(self, model_path, sequences):
        self.__face_classifier = YOLO(model_path)
        self.__face_detector = YOLO("best.pt")
        self.__sequences = sequences

    def track_scores(self, identity, identity_number, ranking, ranking_results):
        # This method keeps track of the scores
        success = 0
        # Since we have 3 rankings for each identity, we determine the cumulative successes for each identity
        for i in range(0, 5):
            # We get the row from the ranking results which is for this identity and for this rank
            row = ranking_results.loc[(ranking_results["identity"] == identity) & (ranking_results["rank-N"] == i)]
            # If the prediction for this rank was for this identity...we have a success!
            if ranking[i] == identity_number:
                success += 1
            if len(row) == 0:
                # If now row was found yet for this identity for this rank...we create it.
                ranking_results.loc[len(ranking_results)] = [identity, i, success, 1]
            else:
                # And otherwise we increment the current success-rate with the success count
                ranking_results.loc[
                    (ranking_results["identity"] == identity) & (ranking_results["rank-N"] == i),
                    "success-rate"] = ranking_results.loc[
                                          (ranking_results["identity"] == identity) & (ranking_results["rank-N"] == i),
                                          "success-rate"] + success
                # And we increment the total number of frames in which the identity was detected by one
                ranking_results.loc[
                    (ranking_results["identity"] == identity) & (ranking_results["rank-N"] == i),
                    "total-frames"] = ranking_results.loc[
                                          (ranking_results["identity"] == identity) & (ranking_results["rank-N"] == i),
                                          "total-frames"] + 1

    def classify(self, training_face_shots:int, test_video: str, visualize) -> pd.DataFrame:
        # In this dataframe we keep track of the number of successes for each identity per rank.
        # Since we have 3 identities, there are 3 rankings per identity.
        ranking_results = pd.DataFrame(columns=["identity", "rank-N", "success-rate", "total-frames"])

        if visualize:
            # We display the test video frames in this window
            cv2.namedWindow("win", cv2.WINDOW_NORMAL)

        cap = cv2.VideoCapture(test_video)

        if not cap.isOpened():
            raise IOError("Error opening the video!")

        # We count the number of frames...
        count = 0
        at_end = False
        while cap.isOpened() & at_end == False:
            ret, img = cap.read()
            if not ret:
                at_end = True
            else:
                # See if we can detect a face...
                face_shot, coordinates = self.get_bounding_boxed_face_shot(img)
                count += 1
                if face_shot is not None:
                    # There was a face detected...now let's predict a classification...
                    results_class = self.__face_classifier.predict(source=face_shot)
                    for result in results_class:
                        probs = result.probs
                        names = result.names
                        # Based on where we are in the test video, we can determine which identity is shown...
                        identity, identity_number = self.determine_identity_in_sequence(test_video, count, names)
                        # We keep track of the scores, which is what this is about...
                        self.track_scores(identity, identity_number, result.probs.top5, ranking_results)
                        if visualize:
                            # We draw a rectangle around the detected face...
                            cv2.rectangle(img, coordinates[0], coordinates[1], (255, 0, 0), 2)
                            # And we display the names of the identities that were classified as the 1st, 2nd and 3rd
                            # including their probabilities
                            top_left_x = coordinates[0][0]
                            top_left_y = coordinates[0][1]
                            for r in range(0,5):
                                txt = str(r + 1) + ":" + str(result.names[probs.top5[r]]) + " (" + str(round(probs.top5conf[r].item(), 4)) + ")"
                                cv2.putText(img, txt, (top_left_x + 5, top_left_y - 5 - 25 * r), self.FONT, 0.6, (255, 255, 255), 2)

                # We also display the current sequence number, the identity that performs in this sequence,
                # the frame count that we are at, and the ranking results so far...
                if visualize:
                    sequence_number = self.determine_sequence_number(test_video, count)
                    cv2.putText(img, "Sequence " + str(sequence_number[0]), (5, 30), self.FONT, 0.6, (255, 255, 0), 2)
                    cv2.putText(img, "Training shots " + str(training_face_shots), (5, 60), self.FONT, 0.6, (255, 255, 0), 2)
                    cv2.putText(img, "Identity " + str(sequence_number[1]), (5, 90), self.FONT, 0.6, (255, 255, 0), 2)
                    cv2.putText(img, "Frame count " + str(count), (5, 120), self.FONT, 0.6, (255, 255, 0), 2)
                    y = 150
                    for row in range(len(ranking_results)):
                        name = ranking_results.loc[row, "identity"]
                        rank_n = ranking_results.loc[row, "rank-N"]
                        successes = ranking_results.loc[row, "success-rate"]
                        total_frames = ranking_results.loc[row, "total-frames"]
                        txt = name + " at rank " + str(rank_n + 1) + " successfully classified " + str(successes) + " out of " + str(total_frames) + " cases"
                        cv2.putText(img, txt, (5, y), self.FONT, 0.6, (255, 255, 0), 2)
                        y += 30
                    cv2.imshow("win", img)
                    key = cv2.waitKey(1)
                    if key == 27:
                        at_end = True

        cap.release()
        return ranking_results

    def get_bounding_boxed_face_shot(self, img):
        # The image is turned into a gray scale image for easier face detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        results = self.__face_detector(img)
        faces = results[0].boxes
        face_shot = None
        coordinates = None
        if faces:
            box = faces[0]
            top_left_x = int(box.xyxy.tolist()[0][0])
            top_left_y = int(box.xyxy.tolist()[0][1])
            bottom_right_x = int(box.xyxy.tolist()[0][2])
            bottom_right_y = int(box.xyxy.tolist()[0][3])
            coordinates = ((top_left_x, top_left_y), (bottom_right_x, bottom_right_y))
            face_shot = img[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
            face_shot = cv2.resize(face_shot, (273, 368), cv2.INTER_AREA)

        return face_shot, coordinates

    def determine_identity_in_sequence(self, test_video, current_frame_count, identities):
        # Determine the sequence we are in, based on the current frame count
        sequence = self.__sequences.loc[
            (self.__sequences["startframe"] <= current_frame_count) &
            (self.__sequences["endframe"] >= current_frame_count) &
            (self.__sequences["video"] == test_video)]
        # Extract the identity in this sequence
        identity = sequence["identity"].item()

        # We loop through the list of identities to determine the right sequence number...
        for i in identities:
            if identity == identities[i]:
                return identity, i
        # The identity wasn't found...
        return None, None

    def determine_sequence_number(self, test_video, current_frame_count) -> (int, str):
        sequence = self.__sequences.loc[
            (self.__sequences["startframe"] <= current_frame_count) &
            (self.__sequences["endframe"] >= current_frame_count) &
            (self.__sequences["video"] == test_video)]

        if len(sequence["sequence"]) == 0:
            return None, None

        return sequence["sequence"].item(), sequence["identity"].item()
