import numpy as np


def should_bin_be_collected(current_fill_level, accumulation_rate):
  if current_fill_level + accumulation_rate >= 100:
    return True


def add_bins_to_collect(binsids, next_collection_day, must_go_bins, current_fill_levels, accumulation_rates, current_collection_day):
  for i in binsids:
    if i in must_go_bins:
      continue
    else:
      for j in range(current_collection_day + 1, next_collection_day):
        if current_fill_levels[i] + j * accumulation_rates[i] >= 100:
          must_go_bins.append(i)
          break
  return must_go_bins


def update_fill_levels_after_first_collection(binsids, must_go_bins, current_fill_levels):
  current_fill_levels = current_fill_levels.copy()
  for i in binsids:
    if i in must_go_bins:
      current_fill_levels[i] = 0
  return current_fill_levels


def initialize_lists_of_bins(binsids):
  next_collection_days = []
  for i in range(0, len(binsids)):
    next_collection_days.append(0)
  return next_collection_days


def calculate_next_collection_days(must_go_bins, current_fill_levels, accumulation_rates, binsids):
  next_collection_days = initialize_lists_of_bins(binsids)
  temporary_fill_levels = current_fill_levels.copy()
  for i in must_go_bins:
    current_day = 0
    while temporary_fill_levels[i] < 100:
      temporary_fill_levels[i] = temporary_fill_levels[i] + accumulation_rates[i]
      current_day = current_day + 1
    next_collection_days[i] = current_day # assuming collection happens at the beginning of the day
  return next_collection_days


def get_next_collection_day(must_go_bins, current_fill_levels, accumulation_rates, binsids):
  next_collection_days = calculate_next_collection_days(must_go_bins, current_fill_levels, accumulation_rates, binsids)
  next_collection_days_array = np.array(next_collection_days)
  next_collection_day = np.min(next_collection_days_array[np.nonzero(next_collection_days_array)])
  return next_collection_day
