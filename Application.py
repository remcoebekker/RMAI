import os
import shutil
import pandas as pd
import DatasetCollector
import FaceClassifier
import Trainer
import dataframe_image as dfi
import DataAnalysis
import VisualizeResults
import Validator

# The nine identities to be trained on
TRAINING_IDENTITIES = ["Etienne", "Robbert", "Roos", "Pim", "Stacey", "Marije", "Leonie", "Remco", "Richard"]
# The three mp4 files of the nine identities to be trained on
TRAINING_VIDEOS = ["./Video/BTP_Etienne.mp4", "./Video/BTP_Robbert.mp4", "./Video/BTP_Roos.mp4",
                   "./Video/DS_Pim.mp4", "./Video/DS_Stacey.mp4", "./Video/CX_Marije.mp4",
                   "./Video/AT_Leonie.mp4", "./Video/AT_Remco.mp4", "./Video/AT_Richard.mp4"]
# The various folders that will be used
DATA_FOLDER = "./data"
TRAINING_FOLDER = "/train"
TESTING_FOLDER = "/test"
VALIDATION_FOLDER = "/val"
DATASET_COLLECTION_FOLDERS = [VALIDATION_FOLDER, TESTING_FOLDER, TRAINING_FOLDER]
# The test video on which we test how well the identities are recognized
TEST_VIDEOS = ["./video/AT.mp4", "./video/DS_seminar.mp4", "./video/CX_MC.mp4", "./video/BTP.mp4"]
# Determines how many frames we are sampling from the training videos
NUMBER_TRAINING_FACE_SHOTS_TESTED = [1, 2, 3, 4, 5, 6, 7, 8, 9]
# The number of face shots collected for validation, testing and training respectively
NUMBER_DATASET_COLLECTION_FACE_SHOTS = [7, 3, 9]
# The base path in which the trained model can be found
MODEL_BASE_PATH = "runs/classify/"


def run(copy_files: bool, retrain_model: bool, visualize: bool):
    """
    Main function for the overall flow of the application.
    """

    # First we set-up the training data, if needed
    set_up_training_data(copy_files)

    # Next step is training the models for the different number of face shots...
    if retrain_model:
        retrain_models()
        validator = Validator.Validator()
        validator.validate_or_test(MODEL_BASE_PATH, NUMBER_TRAINING_FACE_SHOTS_TESTED, "val")
        validator.validate_or_test(MODEL_BASE_PATH, NUMBER_TRAINING_FACE_SHOTS_TESTED, "test")


    # The classification results are assembled in one data frame...
    ranking_results = pd.DataFrame(columns=["identity", "rank-N", "success-rate", "total-frames",
                                            "identification-rate", "training_face_shots"])

    # Now we are ready to put the trained models to the test
    # For each number of training shots, we have a separately trained model and run it on the test video!
    print("We are testing on a test video...please wait a few minutes while we are initializing")
    ranking_results = test_classification(ranking_results, visualize)
    # A sample of the ranking results is exported as an image
    dfi.export(ranking_results, "ranking_results.png", max_rows=10)
    # The ranking results data frame are saved as a csv file
    ranking_results.to_csv("ranking_results.csv")
    # The rankings results are visualized in a facet grid and in a bar plot
    VisualizeResults.generate_facet_grid(NUMBER_TRAINING_FACE_SHOTS_TESTED)
    VisualizeResults.generate_bar_plot()
    # The data analysis tables are generated showing the comparisons between different groups of identities
    # These are the combinations of groups that we originally want to compare
    DataAnalysis.generate_comparison_table("Comparison 1.png",
                                           [["Men without facial hair", "Men with facial hair", "Women"],
                                            ["Men without facial hair", "Men with facial hair"],
                                            ["Men without facial hair", "Women"],
                                            ["Men with facial hair", "Women"]])

    # This is the combination we wanted to look into in the discussion area
    DataAnalysis.generate_comparison_table("Comparison 2.png", [["Men", "Women"]])


def test_classification(ranking_results, visualize) -> pd.DataFrame:
    # We loop through the different number of training face shots...
    for i in range(0, len(NUMBER_TRAINING_FACE_SHOTS_TESTED)):
        # For each number of training face shots, there is a different trained model
        model_path = MODEL_BASE_PATH + "face_shots_" + str(NUMBER_TRAINING_FACE_SHOTS_TESTED[i]) + "/weights/best.pt"
        # And we classify all the identities in the test video...
        face_classifier = FaceClassifier.FaceClassifier(model_path, get_sequences())
        for j in range(0, len(TEST_VIDEOS)):
            classification_results = face_classifier.classify(NUMBER_TRAINING_FACE_SHOTS_TESTED[i],
                                                              TEST_VIDEOS[j],
                                                              visualize)
            # We add a new column to the returned data frame which is a proportion number
            classification_results["identification-rate"] = classification_results["success-rate"] / \
                                                            classification_results["total-frames"]
            # And we add the column of the number of training face shots that was used
            classification_results["training_face_shots"] = NUMBER_TRAINING_FACE_SHOTS_TESTED[i]
            # We assemble all results in one data frame
            ranking_results = pd.concat([ranking_results, classification_results])

    ranking_results["rank-N"] = ranking_results["rank-N"] + 1
    return ranking_results


def retrain_models():
    # For each number of face shots we train a separate model...
    trainer = Trainer.Trainer()
    for i in range(0, len(NUMBER_TRAINING_FACE_SHOTS_TESTED)):
        source_base_folder = DATA_FOLDER + "_" + str(NUMBER_TRAINING_FACE_SHOTS_TESTED[i])
        trainer.train(source_base_folder, "face_shots_" + str(NUMBER_TRAINING_FACE_SHOTS_TESTED[i]))


def set_up_training_data(copy_files):
    # We instantiate a Dataset collector object which will extract face shots from the training videos
    collector = DatasetCollector.DatasetCollector()
    # We check whether the face shots are already collected from the videos
    target_folder = DATA_FOLDER + "_" + str(NUMBER_TRAINING_FACE_SHOTS_TESTED[-1]) + TRAINING_FOLDER
    if not DatasetCollector.are_training_faces_already_collected_from_videos(target_folder,
                                                                             TRAINING_IDENTITIES,
                                                                             NUMBER_TRAINING_FACE_SHOTS_TESTED[-1]):
        # This is not the case, so we collect the faces from the videos
        print("We need to collect the faces from the training videos...this will take a few minutes")
        # We create the data folder that holds the maximum number of training face shots
        target_base_folder = DATA_FOLDER + "_" + str(NUMBER_TRAINING_FACE_SHOTS_TESTED[-1])
        os.mkdir(target_base_folder)
        for i in range(0, len(DATASET_COLLECTION_FOLDERS)):
            target_folder = target_base_folder + DATASET_COLLECTION_FOLDERS[i]
            print("Creating and populating data folder", target_folder)
            os.mkdir(target_folder)
            collector.collect_faces_from_videos(target_folder,
                                                TRAINING_VIDEOS,
                                                TRAINING_IDENTITIES,
                                                NUMBER_DATASET_COLLECTION_FACE_SHOTS[i])

    # The data folder with the maximum number of training face shots has been set up. Now we populate the other
    # folders with less training face shots. We base these folders on the data folder with the maximum number of
    # training face shots
    if copy_files:
        source_base_folder = DATA_FOLDER + "_" + str(NUMBER_TRAINING_FACE_SHOTS_TESTED[-1])
        for i in range(0, len(NUMBER_TRAINING_FACE_SHOTS_TESTED) - 1):
            target_base_folder = DATA_FOLDER + "_" + str(NUMBER_TRAINING_FACE_SHOTS_TESTED[i])
            os.mkdir(target_base_folder)
            for j in range(0, len(DATASET_COLLECTION_FOLDERS)):
                source_folder = source_base_folder + DATASET_COLLECTION_FOLDERS[j]
                target_folder = target_base_folder + DATASET_COLLECTION_FOLDERS[j]
                print(target_folder)
                os.mkdir(target_folder)
                for k in range(0, len(TRAINING_IDENTITIES)):
                    if DATASET_COLLECTION_FOLDERS[j] == TRAINING_FOLDER:
                        files = list(filter(
                            lambda x: int(x.split(".")[2]) < NUMBER_TRAINING_FACE_SHOTS_TESTED[i],
                            os.listdir(source_folder + "/" + TRAINING_IDENTITIES[k])))
                    else:
                        files = os.listdir(source_folder + "/" + TRAINING_IDENTITIES[k])

                    print(files)
                    os.mkdir(target_folder + "/" + TRAINING_IDENTITIES[k])
                    for file in range(0, len(files)):
                        shutil.copyfile(source_folder + "/" + TRAINING_IDENTITIES[k] + "/" + files[file],
                                        target_folder + "/" + TRAINING_IDENTITIES[k] + "/" + files[file])


def get_sequences():
    # The 3 different identities appear in different sequences of the test video. In order to keep track of the
    # right classification, we need to know in which frame which identity appears. The following rows represent this
    # information
    sequences = pd.DataFrame(columns=["video", "sequence", "startframe", "endframe", "identity"])
    sequences.loc[len(sequences)] = [TEST_VIDEOS[0], 1, 1, 934, "Remco"]
    sequences.loc[len(sequences)] = [TEST_VIDEOS[0], 2, 935, 2228, "Leonie"]
    sequences.loc[len(sequences)] = [TEST_VIDEOS[0], 3, 2229, 3492, "Richard"]
    # At the end there all black frames...in which no identity appears...
    sequences.loc[len(sequences)] = [TEST_VIDEOS[0], 4, 3493, 4000, "Blank"]
    sequences.loc[len(sequences)] = [TEST_VIDEOS[1], 1, 1, 218, "Pim"]
    sequences.loc[len(sequences)] = [TEST_VIDEOS[1], 2, 219, 313, "Stacey"]
    sequences.loc[len(sequences)] = [TEST_VIDEOS[1], 3, 314, 430, "Pim"]
    sequences.loc[len(sequences)] = [TEST_VIDEOS[1], 4, 431, 563, "Stacey"]
    sequences.loc[len(sequences)] = [TEST_VIDEOS[1], 5, 564, 721, "Blank"]
    sequences.loc[len(sequences)] = [TEST_VIDEOS[2], 1, 1, 1058, "Marije"]
    sequences.loc[len(sequences)] = [TEST_VIDEOS[2], 2, 1059, 1251, "Blank"]
    sequences.loc[len(sequences)] = [TEST_VIDEOS[3], 1, 1, 179, "Roos"]
    sequences.loc[len(sequences)] = [TEST_VIDEOS[3], 2, 180, 446, "Robbert"]
    sequences.loc[len(sequences)] = [TEST_VIDEOS[3], 3, 447, 658, "Etienne"]
    sequences.loc[len(sequences)] = [TEST_VIDEOS[3], 4, 659, 1013, "Roos"]
    sequences.loc[len(sequences)] = [TEST_VIDEOS[3], 5, 1014, 1329, "Robbert"]
    sequences.loc[len(sequences)] = [TEST_VIDEOS[3], 6, 1330, 1578, "Etienne"]
    sequences.loc[len(sequences)] = [TEST_VIDEOS[3], 7, 1579, 1903, "Roos"]
    sequences.loc[len(sequences)] = [TEST_VIDEOS[3], 8, 1904, 2082, "Robbert"]
    sequences.loc[len(sequences)] = [TEST_VIDEOS[3], 9, 2083, 2110, "Etienne"]
    sequences.loc[len(sequences)] = [TEST_VIDEOS[3], 10, 2111, 2146, "Robbert"]
    sequences.loc[len(sequences)] = [TEST_VIDEOS[3], 11, 2147, 2222, "Roos"]
    sequences.loc[len(sequences)] = [TEST_VIDEOS[3], 12, 2223, 2493, "Blank"]
    return sequences


# If this module is run, it will call the run function
if __name__ == "__main__":
    # The application is kicked off with 3 boolean parameters:
    # - Whether the folders with the face shots used for training will need to be set up
    # - Whether the models need to be trained
    # - Whether the tests of the face recognition need to be visualized
    run(True, True, True)
