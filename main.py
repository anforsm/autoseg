def fibonacci(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n - 1) + fibonacci(
            n - 2
        )  # this is a super long comment that should hopefully get cought by the black formatter. Hopefully now the black pre commit hook catches it


if __name__ == "__main__":
    print(fibonacci(10))
