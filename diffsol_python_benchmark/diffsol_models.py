def robertson_ode(ngroups: int) -> str:
    u_i = (
        f'(0:{ngroups}): x = 1,\n'
        f'({ngroups}:{2 * ngroups}): y = 0,\n'
        f'({2 * ngroups}:{3 * ngroups}): z = 0,\n'
    )
    code = """
        k1 { 0.04 }
        k2 { 30000000 }
        k3 { 10000 }
        u_i {
        """ + u_i + """
        }
        F_i {
            -k1 * x_i + k3 * y_i * z_i,
            k1 * x_i - k2 * y_i * y_i - k3 * y_i * z_i,
            k2 * y_i * y_i,
        }
        """
    return code

def robertson(rtol: float, atol: float) -> str:
    return """
        in = [k1, k2, k3]
        k1 { 0.04 }
        k2 { 10000 }
        k3 { 30000000 }
        u_i {
            x = 1,
            y = 0,
            z = 0,
        }
        dudt_i {
            dxdt = 1,
            dydt = 0,
            dzdt = 0,
        }
        M_i {
            dxdt,
            dydt,
            0,
        }
        F_i {
            -k1 * x + k2 * y * z,
            k1 * x - k2 * y * z - k3 * y * y,
            1 - x - y - z,
        }
        out_i {
            x,
            y,
            z,
        }
        """
