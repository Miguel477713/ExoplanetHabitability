import numpy
import pandas
import jax


def buildFeaturesWithMedianImputation(dataFrame, numericColumns):
    newColumnsToAdd: dict[str, numpy.ndarray] = {}
    imputedColumnCount = 0
    totalImputedValues = 0

    for column in numericColumns:
        colValues = dataFrame[column].to_numpy(dtype=numpy.float64, copy=False)
        missingMask = numpy.isnan(colValues)
        missingCount = int(missingMask.sum())

        if missingCount == 0:
            continue

        imputedColumnCount += 1
        totalImputedValues += missingCount

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

    return dataFrame, {
        "numeric_columns_imputed": int(imputedColumnCount),
        "numeric_values_imputed": int(totalImputedValues),
        "indicator_columns_added": int(len(newColumnsToAdd)),
    }



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

    summary = {
        "rows_before_preprocessing": int(dataFrame.shape[0]),
        "columns_before_preprocessing": int(dataFrame.shape[1]),
        "missing_values_before_preprocessing": int(dataFrame.isna().sum().sum()),
        "description_columns_requested_for_drop": int(len(descriptionColumns)),
        "categorical_column_count_before_encoding": int(len(categorical)),
    }

    errorColumns = list(dataFrame.filter(like="_ERROR_").columns)
    limitColumns = list(dataFrame.filter(like="_LIMIT").columns)
    hzColumns = list(dataFrame.filter(like="_HZ_").columns)
    minColumns = list(dataFrame.filter(like="_MIN").columns)
    maxColumns = list(dataFrame.filter(like="_MAX").columns)
    removableDescriptionColumns = [column for column in descriptionColumns if column in dataFrame.columns]

    summary["error_columns_removed"] = int(len(errorColumns))
    summary["limit_columns_removed"] = int(len(limitColumns))
    summary["hz_columns_removed"] = int(len(hzColumns))
    summary["min_columns_removed"] = int(len(minColumns))
    summary["max_columns_removed"] = int(len(maxColumns))
    summary["description_columns_removed"] = int(len(removableDescriptionColumns))

    dataFrame = (
        dataFrame.drop(columns=errorColumns)
        .drop(columns=limitColumns)
        .drop(columns=hzColumns)
        .drop(columns=minColumns, errors="ignore")
        .drop(columns=maxColumns, errors="ignore")
        .drop(columns=removableDescriptionColumns, errors="ignore")
    )

    summary["columns_after_column_filtering"] = int(dataFrame.shape[1])

    numericColumns: list[str] = []
    for column in list(dataFrame.columns):
        if column == "P_HABITABLE":
            continue
        if column in categorical:
            continue
        numericColumns.append(column)

    summary["numeric_feature_count_before_imputation"] = int(len(numericColumns))

    dataFrame, imputationSummary = buildFeaturesWithMedianImputation(dataFrame=dataFrame, numericColumns=numericColumns)
    summary.update(imputationSummary)

    missingCategoricalBeforeFill = 0
    for column in categorical:
        if column in dataFrame.columns:
            missingCategoricalBeforeFill += int(dataFrame[column].isna().sum())
    summary["categorical_missing_values_filled"] = int(missingCategoricalBeforeFill)

    existingCategorical = [column for column in categorical if column in dataFrame.columns]
    if existingCategorical:
        dataFrame[existingCategorical] = dataFrame[existingCategorical].fillna("UNKNOWN")
        dataFrame = pandas.get_dummies(
            dataFrame,
            columns=existingCategorical,
            drop_first=True,
            dtype=float,
        )

    summary["columns_after_encoding_and_imputation"] = int(dataFrame.shape[1])
    summary["missing_values_after_preprocessing"] = int(dataFrame.isna().sum().sum())

    Y = jax.numpy.array(dataFrame[["P_HABITABLE"]].to_numpy(dtype=int))
    X = jax.numpy.array(
        dataFrame.drop(columns=["P_HABITABLE"]).to_numpy(dtype=jax.numpy.float64)
    )

    X_df = dataFrame.drop(columns=["P_HABITABLE"])
    featureNames = list(X_df.columns)
    X_df.to_csv("X_features.csv", index=False)

    summary["rows_after_preprocessing"] = int(X.shape[0])
    summary["feature_count_after_preprocessing"] = int(X.shape[1])

    return X, Y, featureNames, summary
