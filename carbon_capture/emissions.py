from pathlib import Path

import pandas as pd
import sisepuede.core.support_classes as sc
import sisepuede.manager.sisepuede_examples as sxl
import sisepuede.manager.sisepuede_file_structure as sfs
import sisepuede.models.afolu as mafl
import sisepuede.utilities._toolbox as sf
from sisepuede.core.model_variable import ModelVariable

_MODVAR_NAME_EF_CONVERSION = "Land Use Conversion Emission Factor"
_MODVAR_NAME_SF_FOREST = "Forest Sequestration Emission Factor"
_MODVAR_NAME_SF_FOREST_YOUNG = "Young Secondary Forest Sequestration Emission Factor"
_MODVAR_NAME_SF_LAND_USE = "Land Use Biomass Sequestration Factor"
_MODVARS_ALL_FROM_DB = [
    _MODVAR_NAME_EF_CONVERSION,
    _MODVAR_NAME_SF_FOREST,
    _MODVAR_NAME_SF_FOREST_YOUNG,
    _MODVAR_NAME_SF_LAND_USE,
]

_DATASET_NAME_TRANSITIONS = "transitions"

_DATASET_NAME_AREAS = "areas"
_UNIT_DEF_AREAS = "ha"
_DICT_UNITS_DEF = {
    _DATASET_NAME_AREAS: _UNIT_DEF_AREAS,
}

_DATASET_NAME_AREA_FRACTIONS = "areas_frac"


def get_vars_from_dbdir(
    path_data: Path,
    modvars: list[str],
) -> pd.DataFrame:
    df_out = None
    for modvar in modvars:
        path_read = path_data.joinpath(f"{modvar}.csv")
        df_cur = pd.read_csv(
            path_read,
        )

        df_out = df_cur if df_out is None else df_out.merge(df_cur, how="inner")

    if df_out is None:
        err = f"Unable to read any of the modvars {modvars} from {path_data}"
        raise ValueError(err)

    return df_out


def clean_area_total(
    df_in: pd.DataFrame,
    model_afolu: mafl.AFOLU,
    time_periods: sc.TimePeriods,
    field_cat: str = "label",
    units_input: str = "ha",
) -> pd.DataFrame:
    """Fix the input data for initial areas"""
    modvar = model_afolu.model_attributes.get_variable(
        model_afolu.model_socioeconomic.modvar_gnrl_area,
    )

    if modvar is None:
        err = "Unable to find modvar_gnrl_area"
        raise ValueError(err)

    # some field info
    all_cats = df_in[field_cat].unique()

    fields = modvar.fields
    if fields is None or len(fields) == 0:
        err = "Unable to find fields in modvar_gnrl_area"
        raise ValueError(err)

    field = fields[0]

    # map to data frame
    df_out = (
        df_in.copy()
        .set_index(
            [field_cat],
        )
        .transpose()
        .reset_index(
            names=time_periods.field_time_period,
        )
        .rename_axis(
            None,
            axis=1,
        )
    )

    df_out[field] = df_out[all_cats].sum(
        axis=1,
    )

    df_out = modvar.get_from_dataframe(
        df_out,
        expand_to_all_categories=True,
        extraction_logic="any_fill",
        fields_additional=[time_periods.field_time_period],
        fill_value=0.0,
    )

    if not isinstance(df_out, pd.DataFrame):
        err = "Unable to find fields in modvar_gnrl_area"
        raise TypeError(err)

    # finally, rescale
    return model_afolu.model_attributes.rescale_fields_to_target(
        df_out,
        [field],
        modvar,
        {
            "area": (units_input, 1),
        },
    )


def clean_areas_frac(
    df_in: pd.DataFrame,
    model_afolu: mafl.AFOLU,
    time_periods: sc.TimePeriods,
    field_cat: str = "label",
) -> pd.DataFrame:
    """Fix the input data for initial areas"""

    matt = model_afolu.model_attributes
    modvar = matt.get_variable(
        model_afolu.modvar_lndu_initial_frac,
    )

    if modvar is None:
        err = "Unable to find modvar_lndu_initial_frac"
        raise ValueError(err)

    dict_repl = matt.get_category_replacement_field_dict(
        modvar,
    )

    # map to data frame
    df_out = (
        df_in.copy()
        .set_index(
            [field_cat],
        )
        .transpose()
        .reset_index(
            names=time_periods.field_time_period,
        )
        .rename_axis(
            None,
            axis=1,
        )
        .rename(
            columns=dict_repl,
        )
    )

    out = modvar.get_from_dataframe(
        df_out,
        expand_to_all_categories=True,
        extraction_logic="any_fill",
        fields_additional=[time_periods.field_time_period],
        fill_value=0.0,
    )

    if not isinstance(out, pd.DataFrame):
        err = "Unable to find fields in modvar_lndu_initial_frac"
        raise TypeError(err)

    return out


def build_inputs_to_overwrite(
    dict_data: dict[str, pd.DataFrame],
    model_afolu: mafl.AFOLU,
    time_periods: sc.TimePeriods,
    units_dict: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Clean and combine the DataFrames to build inputs that overwrite
    example data.
    """

    units_dict = _DICT_UNITS_DEF if not isinstance(units_dict, dict) else units_dict

    ##  GET COMPONENT DFs

    df_areas_inital = clean_areas_frac(
        dict_data[_DATASET_NAME_AREA_FRACTIONS],
        model_afolu,
        time_periods,
    )

    df_area_total = clean_area_total(
        dict_data[_DATASET_NAME_AREAS],
        model_afolu,
        time_periods,
        units_input=units_dict.get(
            _DATASET_NAME_AREAS,
            _UNIT_DEF_AREAS,
        ),
    )

    df_transitions = dict_data.get(
        _DATASET_NAME_TRANSITIONS,
    )

    if df_transitions is None:
        err = f"Unable to find {dict_data} in {dict_data}"
        raise ValueError(err)

    ##  COMBINE

    df_out = sf.merge_output_df_list(
        [df_areas_inital, df_area_total, df_transitions],
        model_afolu.model_attributes,
    ).ffill()

    df_out[time_periods.field_time_period] = df_out[
        time_periods.field_time_period
    ].astype(int)

    return df_out


def build_dataset(
    examples: sxl.SISEPUEDEExamples,
    region: str | int,
    model_afolu: mafl.AFOLU,
    regions: sc.Regions,
    time_periods: sc.TimePeriods,
    *,
    dict_ursa_data: dict[str, pd.DataFrame],
    path_ssp_data: Path,
    units_dict: dict[str, str] | None = None,
) -> pd.DataFrame:
    # initialize a base data frame
    df_base = examples.input_data_frame.copy()  # type: ignore[reportAttributeAccessIssue]

    # get from CSV and from DB
    df_inputs_ursa = build_inputs_to_overwrite(
        dict_ursa_data,
        model_afolu,
        time_periods,
        units_dict=units_dict,
    )

    # get files from pipeline
    # TEMP
    df_inputs_ssp = get_vars_from_dbdir(
        path_ssp_data,
        _MODVARS_ALL_FROM_DB,
    )

    # get region and filter
    df_inputs_ssp = regions.extract_from_df(
        df_inputs_ssp,
        region,
        regions.field_iso,
    )

    if not isinstance(df_inputs_ssp, pd.DataFrame):
        err = f"Unable to find {region} in {df_inputs_ssp}"
        raise TypeError(err)

    # adjust time periods for now
    field_year = time_periods.field_year
    df_inputs_ssp[field_year] -= df_inputs_ssp[field_year].iloc[0]
    df_inputs_ssp = df_inputs_ssp.rename(
        columns={
            field_year: time_periods.field_time_period,
        },
    )

    ##  SET AGGREGATE INPUTS

    df_in = sf.match_df_to_target_df(
        df_base.drop(columns=[regions.key]),
        df_inputs_ursa,
        [time_periods.field_time_period],
        overwrite_only=False,
        try_interpolate=True,
    )

    df_in = sf.match_df_to_target_df(
        df_in,
        df_inputs_ssp.drop(columns=[regions.field_iso]),
        [time_periods.field_time_period],
        overwrite_only=False,
        try_interpolate=True,
    )

    ##  MAKE SOME CITY-SIZE SPECIFIC ADJUSTMENTS

    matt = model_afolu.model_attributes

    # not deal with HWP
    modvar = matt.get_variable("Initial Industrial Production")

    if modvar is None:
        err = "Unable to find modvar_industrial_production"
        raise ValueError(err)

    df_in[modvar.build_fields(category_restrictions=["paper", "wood"])] = 0

    # ensure lndu_reallocation factor is 0
    modvar_lurf = matt.get_variable("Land Use Yield Reallocation Factor")

    if modvar_lurf is None:
        err = "Unable to find modvar_lurf"
        raise ValueError(err)

    df_in[modvar_lurf.fields] = 0

    return df_in


def get_model_attributes(struct: sfs.SISEPUEDEFileStructure) -> sc.ModelAttributes:
    """Get the model attributes from the file structure"""
    matt = struct.model_attributes

    if matt is None:
        err = "Unable to find model attributes"
        raise ValueError(err)

    return matt


def get_modvar(matt: sc.ModelAttributes, name: str) -> ModelVariable:
    """Get the model variable from the model attributes"""
    modvar = matt.get_variable(name)

    if modvar is None:
        err = f"Unable to find {name} in {matt}"
        raise ValueError(err)

    return modvar


def generate_model_objects() -> tuple[
    sxl.SISEPUEDEExamples,
    sc.ModelAttributes,
    mafl.AFOLU,
    sc.Regions,
    sc.TimePeriods,
]:
    """Generate the model objects"""
    # Initialize SISEPUEDE objects
    examples = sxl.SISEPUEDEExamples()
    file_struct = sfs.SISEPUEDEFileStructure()

    # set the model attributes
    matt = get_model_attributes(file_struct)

    # objects that depend on model attributes
    model_afolu = mafl.AFOLU(matt)
    regions = sc.Regions(matt)
    time_periods = sc.TimePeriods(matt)

    return examples, matt, model_afolu, regions, time_periods


def calculate_emissions(
    areas: pd.DataFrame,
    transitions: pd.DataFrame,
    *,
    iso: str
) -> pd.DataFrame:
    # Initialize SISEPUEDE objects
    examples, _, model_afolu, regions, time_periods = generate_model_objects()

    temp = areas.set_index("label")

    areas_frac = temp.div(temp.sum(axis=0), axis=1).reset_index(names="label")

    # run model
    dict_ursa_data = {
        "areas": areas,
        "areas_frac": areas_frac,
        "transitions": transitions,
    }
    df_in = build_dataset(
        examples,
        iso,
        model_afolu,
        regions,
        time_periods,
        dict_ursa_data=dict_ursa_data,
        path_ssp_data=Path("./data")
        / "input"
        / "sisepuede",
    )

    return model_afolu(
        df_in,
    ).set_index("time_period")