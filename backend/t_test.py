import pingouin as pg


def t_test(data, variable, within):
    for mouse in data.mouse.unique():
        mouse_bool = data.mouse == mouse
        group_1 = data.block_labels == data.block_labels.unique()[0]
        group_2 = data.block_labels == data.block_labels.unique()[1]
        res = pg.ttest(data[mouse_bool & group_1][variable], data[mouse_bool & group_2][variable])
        print(f'\nt-test {mouse} {variable}: p = {res["p-val"].values[0]}')
