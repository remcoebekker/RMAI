import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import labellines


def generate_bar_plot():
    # We read in the results...
    ranking_results = pd.read_csv("ranking_results.csv")
    # Select the rows with Rank-1
    selection = ranking_results.loc[(ranking_results["rank-N"] == 1)]
    # Select the rows with identities for which the identification rate >= 0.95
    selection["count_me_in"] = selection["identification-rate"] >= 0.95
    # For investigative purposes we print out this intermediate result
    print(selection)
    # Select the 3 columns needed
    selection = selection[["training_face_shots", "identity", "count_me_in"]]
    # The results are grouped on the number of training face shots and the other columns are
    # summed up
    selection = selection.groupby(["training_face_shots"]).sum()
    # For investigative purposes we print out the result before visualization
    print(selection)
    # The bar graph is created with the on the x-axis the number of training face shots and on the
    # y-axis the summed up number identities for which identification rate >= 0.95
    fig = sns.barplot(data=selection, x=selection.index, y="count_me_in")
    # We show a horizontal red line at 9 identities
    fig.axhline(y=9, color="red")
    fig.set(xlabel='Number of training face shots', ylabel='Number of identities with identification rate >= 95%')
    plt.yticks(list(range(0, 10)))
    plt.show()


def generate_facet_grid(number_of_training_face_shots_tested: list):
    # We read in the results...
    ranking_results = pd.read_csv("ranking_results.csv")
    # The facet grid is set up with the number of training face shots as the facet, the identity
    # as the color
    g = sns.FacetGrid(ranking_results, col="training_face_shots", hue="identity", col_wrap=3)
    # Each facet is filled in with a line plot with rank-N on the x-axis and identification rate
    # on the y-axis
    g.map(sns.lineplot, "rank-N", "identification-rate").add_legend()
    # 5 ranks are plotted on the x-axis
    g.set(xticks=[1, 2, 3, 4, 5])
    # Set up the axis labels
    g.set_titles("{col_name}")  # use this argument literally
    g.set_axis_labels(x_var="Rank-N", y_var="Identification rate")
    # Set up the title of each facet plot and add a red dashed line at 0.95 identification rate
    ordered_title_list = []
    for i in range(len(number_of_training_face_shots_tested)):
        ordered_title_list.append(str(number_of_training_face_shots_tested[i]) + " training face shots")
    for idx, ax in enumerate(g.axes.flat):
        ax.set_title(ordered_title_list[idx])
        lines = ax.get_lines()
        for line in range(0, len(lines)):
            labellines.labelLine(lines[line], x=1.5, label=lines[line].get_label(), fontsize=7)
        ax.axhline(0.95, ls="--", c="red")
    plt.show()


# If this module is run, it will call the generate comparison table function
if __name__ == "__main__":
    # First we generate the facet grid...
    generate_facet_grid([1, 2, 3, 4, 5, 6, 7, 8, 9])
    # Next we generate the bar plot
    generate_bar_plot()
