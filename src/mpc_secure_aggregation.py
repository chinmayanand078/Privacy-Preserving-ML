from mpyc.runtime import mpc


async def secure_sum_local(values):
    """
    Demo: each party starts with a local integer value;
    we compute the global sum using MPyC.
    Run with:
        python -m mpyc src/mpc_secure_aggregation.py -M3 --no-log
    """
    secint = mpc.SecInt(32)

    await mpc.start()
    a = secint(values[mpc.pid])
    s = await mpc.output(mpc.sum(mpc.input(a)))
    print(f"[Party {mpc.pid}] Secure sum = {s}")
    await mpc.shutdown()


def main():
    local_values = [5, 7, 9]
    mpc.run(secure_sum_local(local_values))


if __name__ == "__main__":
    main()
