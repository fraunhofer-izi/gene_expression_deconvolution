from abc import ABC

def del_if_there(instance, varnames):
    """Deletes all variables in varnames if they exist in instance."""
    for var in varnames:
        try:
            delattr(instance, var)
        except AttributeError:
            pass

def property_deps(*args):
    """

    A function decorator to generate set, get and delete methods for a
    property on which others depend.

    Dependent variables are deleted when the property changes or is deleted.
    The decorated function will be used as getter that is executed once at
    request if there is no stored value and cached until it is deleted or
    set manually.

    Parameters
    ----------
    args : string list
        Variable names of the depending variables.

    Returns
    -------
    dep_property : class
        A class with predifined __get__, __set__ and __delete__ methods.

    Usage
    -----
    class some_class(some_super_class):
        @property_deps('some', 'dependent', 'variables')
        def my_property(self):
            ...
            return value
    """

    class baseClass:
        def __init__(self, func):
            self.getter = func
            self.fname = '_cashed_dep_prop_' + func.__name__
        def __get__(self, instance, owner):
            if instance is None:
                return self
            if self.getter is None:
                raise AttributeError("unreadable attribute")
            if not hasattr(instance, self.fname):
                setattr(instance, self.fname, self.getter(instance))
            return getattr(instance, self.fname)
        def __set__(self, instance, value):
            del_if_there(instance, args)
            setattr(instance, self.fname, value)
        def __delete__(self, instance):
            del_if_there(instance, (self.fname,) + args)

    def dep_property(func):
        clsdict = {'__doc__': func.__doc__}
        theClass = type(func.__name__, (baseClass,), clsdict)
        return theClass(func)

    return dep_property
