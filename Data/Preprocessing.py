import numpy
import pandas
import jax

def buildFeaturesWithMedianImputation(dataFrame, numericColumns):
    newColumnsToAdd: dict[str, numpy.ndarray] = {}

    for column in numericColumns:
        colValues = dataFrame[column].to_numpy(dtype=numpy.float64, copy=False)
        missingMask = numpy.isnan(colValues)
        missingCount = int(missingMask.sum())

        if missingCount == 0:
            continue

        missingRatePercent = missingCount / float(colValues.shape[0])

        observed = colValues[~missingMask]
        if observed.size == 0:
            # entire column missing
            medianValue = 0.0
        else:
            medianValue = float(numpy.median(observed))

        # Impute in-place
        colValuesImputed = colValues.copy()
        colValuesImputed[missingMask] = medianValue
        dataFrame[column] = colValuesImputed.astype(numpy.float32)

    # Append any indicator columns at the end (deterministic way although slow)
    if newColumnsToAdd:
        for name in sorted(newColumnsToAdd.keys()):
            dataFrame[name] = newColumnsToAdd[name]

    return dataFrame


def GetJaxArrays(dataFrame):
    descriptionColumns = ["P_NAME", "P_DETECTION", "P_DISCOVERY_FACILITY", "P_YEAR", "P_UPDATE", "P_MASS_ORIGIN",
                            "S_NAME", "S_NAME_HD", "S_NAME_HIP",#names
                            "S_RA", "S_DEC", "S_RA_STR", "S_DEC_STR", "S_RA_TXT", "S_DEC_TXT", #in-sky position
                            "S_CONSTELLATION", "S_CONSTELLATION_ABR", "S_CONSTELLATION_ENG", #constellation
                            "P_ESI", #may already include many physical features, ESI from 2019 to 2025 changed to highly correlated
                            "S_ABIO_ZONE", #E38 exponent, probably sentinent value
                            "P_DISTANCE", "P_DISTANCE_EFF", #highly correlated, actually the same: P_SEMI_MAJOR_AXIS ≈ P_DISTANCE ≈ P_DISTANCE_EFF
                            "P_HABZONE_OPT", "P_HABZONE_CON", # same as P_HABITABLE
                            "S_TYPE", 
                            "P_TYPE_TEMP"] #strong predictor for linear model
            
    categorical = ["S_TYPE_TEMP","P_TYPE"]

    dataFrame = (
        dataFrame.drop(columns=dataFrame.filter(like="_ERROR_").columns)
        .drop(columns=dataFrame.filter(like="_LIMIT").columns)
        .drop(columns=dataFrame.filter(like="_HZ_").columns)
        .drop(columns=dataFrame.filter(like="_MIN").columns, errors="ignore")
        .drop(columns=dataFrame.filter(like="_MAX").columns, errors="ignore")
        .drop(columns=descriptionColumns)
    )

    numericColumns: list[str] = []
    for column in list(dataFrame.columns):
        if column == "P_HABITABLE":
            continue
        if column in categorical:
            continue
        numericColumns.append(column)

    dataFrame = buildFeaturesWithMedianImputation(dataFrame=dataFrame, numericColumns=numericColumns)

    dataFrame[categorical] = dataFrame[categorical].fillna("UNKNOWN")
    dataFrame = pandas.get_dummies(
        dataFrame,
        columns=categorical,
        drop_first=True,
        dtype=float,
    )

    Y = jax.numpy.array(dataFrame[["P_HABITABLE"]].to_numpy(dtype=int))
    X = jax.numpy.array(
        dataFrame.drop(columns=["P_HABITABLE"]).to_numpy(dtype=jax.numpy.float64)
    )

    X_df = dataFrame.drop(columns=["P_HABITABLE"])
    featureNames = list(X_df.columns)
    X_df.to_csv("X_features.csv", index=False)

    return X, Y, featureNames
