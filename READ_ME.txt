This document describes the steps that need to be taken to get the software running that has been needed to produce the report.

1. Start-up Anaconda
2. Import in Anaconda the "RMAI.yaml" file which sets up the right environment for the software.
3. Download the the files in the repository and put them in a folder of your own choice, for instance "RMAI".
4. Under this folder create 1 new sub folder called "Video".
5. You have received a link to the videos. Download the videos (13 in total) and store them in the "Video" sub folder.
6. In Anaconda click on the newly imported environment and select "Open with Python".
7. In the Python terminal window navigate towards the folder just created, for instance "RMAI".
8. Type in and run the "import Application" statement.
9. Type in and run the "Application.run(True, True, True)" statement.
10. Now the application first extracts the face shots from the videos (in the "data_9" subfolder you can view the progress).
11. Next the application will train the models (in the "runs" subfolder you can view the progress).
12. Then the application will start testing on the test videos. 
A window will pop-up showing the frames of the test videos and the results of face detection and classification. 
If during this phase you want to stop the application you can click on Esc button. You will have this a number of times, as it only interrupts the current number of training shots (and there are 9 rounds).
13. When the testing in all test videos has ended, the application will show the results but also save all necessary graphs and files. 
14. The application will end by itself. 