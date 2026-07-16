import numpy as np

def quad_transform(x, params, setting="ring"):
    '''
    Quadratic transformation function
    For 'ring' setting, parameters are [a,b,c]: ax^2 + bx + c
    For 'central' setting, parameters are [b,c]: bx + c
    '''
    setting = setting.lower()
    if setting == "ring":
        a,b,c = params
        return a*x**2 + b*x + c
    elif setting == "central":
        b,c = params
        return b*x + c

def inverse_quad_transform(x, y, params, setting="ring"):
    setting = setting.lower()
    if setting == "ring":
        a,b,c = params
        if np.isclose(a,0):
            return (y - c)/b
        
        vert = -b/(2*a)

        if (max(x) < vert and a > 0) or (min(x) > vert and a < 0):
            branch = -1
        elif (max(x) < vert and a < 0) or (min(x) > vert and a > 0):
            branch = +1
        else:
            raise ValueError("Mapping is not monotonic")
        
        disc = b ** 2 - 4*a*(c - y)
        if any(disc < 0):
            raise ValueError("y is not in range")

        return (-b + branch*np.sqrt(disc)) / (2*a)
    elif setting == "central":
        b,c = params
        return (y - c)/b
        

def legendre_quad_transform(x,params,setting="ring"):
    '''
    Quadratic transformation function using Legendre polynomials
    Variable to be transformed is scaled to be within [-1,1]
    For 'ring' setting, parameters are [a,b,c]: aP_2 + bP_1 + c
    For 'central' setting, parameters are [b,c]: bP_1 + c
    '''
    setting = setting.lower()
    xmin = np.min(x)
    xmax = np.max(x)
    
    z = 2*(x - xmin) / (xmax - xmin) - 1
    P1 = z
    P2 = 0.5*(3*z**2 - 1)
    
    if setting == "ring":
        a,b,c = params
        return a*P2 + b*P1 + c
    elif setting == "central":
        b,c = params
        return b*P1 + c

def inverse_legendre_quad_transform(x, y, params, setting="ring"):
    setting = setting.lower()

    xmin = np.min(x)
    xmax = np.max(x)

    if setting == "ring":
        a, b, c = params

        A = 1.5 * a
        B = b
        C = c - 0.5 * a

        if np.isclose(A, 0):
            z = (y - C) / B
        else:
            vert = -B / (2*A)

            if (max(x) < (xmin + (vert+1)*(xmax-xmin)/2) and A > 0) or \
               (min(x) > (xmin + (vert+1)*(xmax-xmin)/2) and A < 0):
                branch = -1
            elif (max(x) < (xmin + (vert+1)*(xmax-xmin)/2) and A < 0) or \
                 (min(x) > (xmin + (vert+1)*(xmax-xmin)/2) and A > 0):
                branch = +1
            else:
                raise ValueError("Mapping is not monotonic")

            disc = B**2 - 4*A*(C - y)

            if np.any(disc < 0):
                raise ValueError("y is not in the range of the quadratic")

            z = (-B + branch*np.sqrt(disc)) / (2*A)

    elif setting == "central":
        b, c = params
        z = (y - c)/b

    return xmin + (z + 1)*(xmax - xmin)/2

def power_transform(x,params,setting="ring"):
    '''
    Transformation function of the form
    - bx^a + c for ring channels (parameters [a,b,c])
    - bx + c for central channels (parameters [b,c])
    '''
    setting = setting.lower()
    if setting == "ring":
        a,b,c = params
        return b*x**a + c
    elif setting == "central":
        b,c = params
        return b*x + c