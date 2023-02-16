import pingouin as pg


def anova(data, variable, within, subject):
    """
    This is not quite right, need to think about when to use the anova specifically, this time I just need a t-test
    cause I want each mouse individually
    """
    res = pg.rm_anova(dv=variable, within=within, subject=subject,
                      data=data, detailed=True)
    print(f'\nanova {variable}:')
    print(res[['Source', 'p-unc']])

    post_hocs = pg.pairwise_ttests(dv=variable, within=[within], subject=subject,
                                   padjust='fdr_bh', data=data)
    print(f'\npost hocs {variable}:')
    print(post_hocs[['A', 'B', 'p-corr']])
    print(post_hocs[['A', 'B', 'p-unc']])
