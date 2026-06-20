import dagster as dg

strain_relief = dg.define_asset_job("all_assets", selection=dg.AssetSelection.all())


@dg.definitions
def jobs():
    return dg.Definitions(jobs=[strain_relief])
