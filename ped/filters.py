'''
Copyright:
Alexander Blinne, 2018
'''


from . import filterfactories

RemoveDeadAndHotPixels = filterfactories.Median(size=2)
