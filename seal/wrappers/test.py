# elif (
#         origin in {collections.abc.Mapping, Mapping, Dict, dict}
#         and len(args) == 2
#         and can_construct_from_params(args[-1])
#     ):
#         value_cls = annotation.__args__[-1]
#         value_dict = {}
#         if not isinstance(popped_params, Mapping):
#             raise TypeError(
#                 f"Expected {argument_name} to be a Mapping (probably a dict or a Params object)."
#             )

#         for key, value_params in popped_params.items():
#             value_dict[key] = construct_arg(
#                 str(value_cls),
#                 argument_name + "." + key,
#                 value_params,
#                 value_cls,
#                 _NO_DEFAULT,
#                 **extras,
#             )

#         return value_dict

# elif origin in (Tuple, tuple) and all(can_construct_from_params(arg) for arg in args):
#         value_list = []

#         for i, (value_cls, value_params) in enumerate(zip(annotation.__args__, popped_params)):
#             value = construct_arg(
#                 str(value_cls),
#                 argument_name + f".{i}",
#                 value_params,
#                 value_cls,
#                 _NO_DEFAULT,
#                 **extras,
#             )
#             value_list.append(value)

#         return tuple(value_list)

# elif origin in (Set, set) and len(args) == 1 and can_construct_from_params(args[0]):
#         value_cls = annotation.__args__[0]

#         value_set = set()

#         for i, value_params in enumerate(popped_params):
#             value = construct_arg(
#                 str(value_cls),
#                 argument_name + f".{i}",
#                 value_params,
#                 value_cls,
#                 _NO_DEFAULT,
#                 **extras,
#             )
#             value_set.add(value)

#         return value_set

#  elif (
#         origin in {collections.abc.Iterable, Iterable, List, list}
#         and len(args) == 1
#         and can_construct_from_params(args[0])
#     ):
#         value_cls = annotation.__args__[0]

#         value_list = []

#         for i, value_params in enumerate(popped_params):
#             value = construct_arg(
#                 str(value_cls),
#                 argument_name + f".{i}",
#                 value_params,
#                 value_cls,
#                 _NO_DEFAULT,
#                 **extras,
#             )
#             value_list.append(value)

#         return value_list