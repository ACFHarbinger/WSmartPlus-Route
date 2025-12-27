# Functions for condition checks

# Check route feasibility
# function to check routes feasibility regarding number of overflowing bins allowed
# the visited number of bins to which stock + accumulation_rate >= capacity has to be greater
# or equal than the number of bins that verify this condition minus the total number of bins
# multiplied by the percentage of bins that can overflow
def check_bins_overflowing_feasibility(data, routes_list, number_of_bins, perc_bins_can_overflow, E, B):
    bins_overflowing = []
    for index, row in data.iterrows():
        if row['Stock'] + row['Accum_Rate'] >= E * B:
            bins_overflowing.append(row['#bin'])
    
    number_bins_overflowing = len(bins_overflowing)
    check = number_bins_overflowing - number_of_bins * perc_bins_can_overflow
    overflowing_bins_in_routes = []
    for i in bins_overflowing:
        if any(i in a for a in routes_list):
            overflowing_bins_in_routes.append(1)
        else:
            overflowing_bins_in_routes.append(0)

    total_ovf_bins_in_routes = sum(overflowing_bins_in_routes)
    if total_ovf_bins_in_routes >= check:
        status = 'pass'
    else:
        status = 'fail'
    return status


# Check route admissibility
# checks if every function is working fine and admissible solutions are being generated
def check_solution_admissibility(routes_list, removed_bins, number_of_bins):
    total_length = 0
    for i in routes_list:
        total_length += len(i)
    if (total_length + len(removed_bins)) == number_of_bins + (2 * len(routes_list)):
        admissible = True
    else:
        admissible = False
    return admissible