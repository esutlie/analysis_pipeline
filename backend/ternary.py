def ternary(n):
    if n == 0:
        return '0'
    nums = []
    while n:
        n, r = divmod(n, 3)
        nums.append(str(r))
    return ''.join(reversed(nums))


def to_ternary(s):
    num = 0
    for digit, val in enumerate(s[::-1]):
        if int(val) > 2:
            print('digit greater than 2 encountered')
            return None
        num += (3 ** digit) * int(val)
    return num


if __name__ == '__main__':
    print(ternary(50))
    print(to_ternary('1201'))
