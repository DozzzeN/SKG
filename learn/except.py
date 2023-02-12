def foo(s):
    return 10 / int(s)


def bar(s):
    return foo(s) * 2


bar('0')

# try:
#     bar('0')
# except Exception as e:
#     print('Error:', e)
# finally:
#     print('finally...')

# try:
#     print('try...')
#     r = 10 / int('0')
#     print('result:', r)
# except ValueError as e:
#     print('ValueError:', e)
# except ZeroDivisionError as e:
#     print('ZeroDivisionError:', e)
# else:
#     print('no error!')
# finally:
#     print('finally...')
# print('END')
