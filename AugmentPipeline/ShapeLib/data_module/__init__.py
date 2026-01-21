from typing import TypeAlias

DatasetName = str
AVAILABLE_JAW_SOURCES: list[DatasetName] = ['OOD', '3DS']

ToothLabelFDI = str
VALID_FDI_LABELS_UPPER: list[ToothLabelFDI] = ['11', '12', '13', '14', '15', '16', '17', '18',
                                               '21', '22', '23', '24', '25', '26', '27', '28']
VALID_FDI_LABELS_LOWER: list[ToothLabelFDI] = ['31', '32', '33', '34', '35', '36', '37', '38',
                                               '41', '42', '43', '44', '45', '46', '47', '48']

ToothInfo = dict[ToothLabelFDI, dict]
JawInfo = dict[str, dict]  # maxilla/mandible, pre/post-operative, etc.