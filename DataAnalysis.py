import pandas as pd
from scipy import stats
import dataframe_image as dfi


def determine_background_color(cell_value):
    # Default background color is white
    color = "white"
    if isinstance(cell_value, float):
        if cell_value >= 0.05:
            # The cell value is not significant, so we make the background green
            color = '#C1E1C1'
    else:
        # The cell is textual (row and column headers), so background is off-white
        color = "#FAF9F6"
    return f'background-color: {color}'


def add_group(ranking_results: pd.DataFrame, groupings: list):
    ranking_results["group"] = "Women"
    men_without_facial_hair = ["Remco", "Etienne"]
    men_with_facial_hair = ["Richard", "Pim", "Robbert"]
    men = men_without_facial_hair + men_with_facial_hair
    if "Men" in groupings[0]:
        ranking_results.loc[(ranking_results["identity"].isin(men)), "group"] = "Men"
    else:
        ranking_results.loc[(ranking_results["identity"].isin(men_with_facial_hair)), "group"] = "Men with facial hair"
        ranking_results.loc[(ranking_results["identity"].isin(men_without_facial_hair)), "group"] = "Men without facial hair"

    return ranking_results


def prepare_data_frame(ranking_results, groups, face_shots):
    # We select the Rank-1 rows, where the number of training face shots is equal to the given number
    # and we only select the desired groups of identities
    selection = ranking_results.loc[(ranking_results["rank-N"] == 1) &
                                    (ranking_results["training_face_shots"] == face_shots) &
                                    (ranking_results["group"].isin(groups))]
    # We only need these three columns
    selection = selection[["group", "success-rate", "total-frames"]]
    # We sum up the success rates and the number of total frames per group
    selection = selection.groupby(["group"]).sum()
    # And we determine the overall number of total frames and number of total successes
    total_frames = selection["total-frames"].sum()
    total_successes = selection["success-rate"].sum()
    # Based on these we determine the expected number of successes based on the number of total frames per group
    selection["expected"] = (total_successes / total_frames) * selection["total-frames"]
    return selection


def generate_comparison_table(filename: str, groupings: list):
    # We read in the ranking results...
    ranking_results = pd.read_csv("ranking_results.csv")
    # This data frame is the basis for the comparison table we want to generate
    p_values = pd.DataFrame(columns=["Comparison",
                                     "1 face shot",
                                     "2 face shots",
                                     "3 face shots",
                                     "4 face shots",
                                     "5 face shots",
                                     "6 face shots",
                                     "7 face shots",
                                     "8 face shots",
                                     "9 face shots"])
    # We add the group to which a identity belongs to the rankings results data
    ranking_results = add_group(ranking_results, groupings)
    # We select the columns we need
    ranking_results = ranking_results[["rank-N", "training_face_shots", "group", "success-rate", "total-frames"]]
    # We loop through the combinations of groups we want to compare...
    for grouping in range(0, len(groupings)):
        # The row header is put together from the groups in the combination
        comparison = groupings[grouping][0]
        for i in range(1, len(groupings[grouping])):
            comparison = comparison + " vs. " + groupings[grouping][i]
        # The row in the data frame starts with the row header
        p_value_row = [comparison]
        # And next the p-values is determined for each number of face shots investigated
        for face_shots in range(1, 10):
            # A selection of the raw results is made
            selection = prepare_data_frame(ranking_results, groupings[grouping], face_shots)
            # We export the selection for investigative purposes
            dfi.export(selection, comparison + "-" + str(face_shots) + ".png", max_rows=10)
            # A chisquare test is performed on the combination of groups to determine whether the group could be
            # of influence on the number of correct identifications
            results = stats.chisquare(selection["success-rate"], selection["expected"])
            # We add the p-value to the row
            if round(results.pvalue,4) == 0:
                p_value_row.append("<.0001")
            else:
                p_value_row.append(round(results.pvalue, 4))

        # The row is added to the data frame we are building
        p_values.loc[len(p_values)] = p_value_row
    # We create a styler object for the data frame for correctly formatting the p-values: suppressing the leading zero
    # and showing 4 decimal places
    styler = p_values.style.format(lambda s: ("%.4f" % s).replace("-0", "-").lstrip("0") if isinstance(s, float) else s)
    styler.map(determine_background_color).hide()
    styler.set_table_styles(
        [{
            'selector': 'th',
            'props': [
                ('background-color', '#40826D'),
                ('color', 'white')]
        }])
    # Export the formatted table as an image
    dfi.export(styler, filename)


# If this module is run, it will call the generate comparison table function
if __name__ == "__main__":
    # These are the combinations of groups that we originally want to compare
    generate_comparison_table("Comparison 1.png", [["Men without facial hair", "Men with facial hair", "Women"],
                                                   ["Men without facial hair", "Men with facial hair"],
                                                   ["Men without facial hair", "Women"],
                                                   ["Men with facial hair", "Women"]])

    # This is the combination we wanted to look into in the discussion area
    generate_comparison_table("Comparison 2.png", [["Men", "Women"]])
