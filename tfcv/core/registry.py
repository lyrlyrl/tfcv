"""Registry utility."""

def register(registered_collection, reg_key):
    """Register decorated function or class to collection.

    Register decorated function or class into registered_collection, in a
    hierarchical order. For example, when reg_key="my_model/my_exp/my_config_0"
    the decorated function or class is stored under
    registered_collection["my_model"]["my_exp"]["my_config_0"].
    This decorator is supposed to be used together with the lookup() function in
    this file.

    Args:
        registered_collection: a dictionary. The decorated function or class will be
            put into this collection.
        reg_key: The key for retrieving the registered function or class. If reg_key
            is a string, it can be hierarchical like my_model/my_exp/my_config_0
    Returns:
        A decorator function
    Raises:
        KeyError: when function or class to register already exists.
    """
    def decorator(fn_or_cls):
        """Put fn_or_cls in the dictionary."""
        if isinstance(reg_key, str):
            hierarchy = reg_key.split("/")
            collection = registered_collection
            for h_idx, entry_name in enumerate(hierarchy[:-1]):
                if entry_name not in collection:
                    collection[entry_name] = {}
                collection = collection[entry_name]
                if not isinstance(collection, dict):
                    raise KeyError(
                            "Collection path {} at position {} already registered as "
                            "a function or class.".format(entry_name, h_idx))
            leaf_reg_key = hierarchy[-1]
        else:
            collection = registered_collection
            leaf_reg_key = reg_key

        if leaf_reg_key in collection:
            raise KeyError("Function or class {} registered multiple times.".format(
                    leaf_reg_key))
        collection[leaf_reg_key] = fn_or_cls
        return fn_or_cls
    return decorator


def lookup(registered_collection, reg_key):
    """Lookup and return decorated function or class in the collection.

    Lookup decorated function or class in registered_collection, in a
    hierarchical order. For example, when
    reg_key="my_model/my_exp/my_config_0",
    this function will return
    registered_collection["my_model"]["my_exp"]["my_config_0"].

    Args:
        registered_collection: a dictionary. The decorated function or class will be
            retrieved from this collection.
        reg_key: The key for retrieving the registered function or class. If reg_key
            is a string, it can be hierarchical like my_model/my_exp/my_config_0
    Returns:
        The registered function or class.
    Raises:
        LookupError: when reg_key cannot be found.
    """
    if isinstance(reg_key, str):
        hierarchy = reg_key.split("/")
        collection = registered_collection
        for h_idx, entry_name in enumerate(hierarchy):
            if entry_name not in collection:
                raise LookupError(
                        "collection path {} at position {} never registered.".format(
                                entry_name, h_idx))
            collection = collection[entry_name]
        return collection
    else:
        if reg_key not in registered_collection:
            raise LookupError("registration key {} never registered.".format(reg_key))
        return registered_collection[reg_key]