# Copyright 2008-2019 pydicom authors. See LICENSE file for details.
"""Access code dictionary information"""

from itertools import chain
import inspect
from typing import cast, Union
from collections.abc import KeysView, Iterable

from pydicom.sr.coding import Code
from pydicom.sr._concepts_dict import concepts as CONCEPTS
from pydicom.sr._cid_dict import name_for_cid, cid_concepts as CID_CONCEPTS


# Reverse lookup for cid names
cid_for_name = {v: k for k, v in name_for_cid.items()}


def _filtered(source: Iterable[str], filters: Iterable[str]) -> list[str]:
    """Return a sorted list of filtered str.

    Parameters
    ----------
    source : Iterable[str]
        The iterable of str to be filtered.
    filters : Iterable[str]
        An iterable containing patterns for which values are to be included
        in the results.

    Returns
    -------
    List[str]
        A sorted list of unique values from `source`, filtered by including
        case-insensitive partial or full matches against the values
        in `filters`.
    """
    if not filters:
        return sorted(set(source))

    filters = [f.lower() for f in filters]

    return sorted(
        set(val for val in source if any((f in val.lower()) for f in filters))
    )


ConceptsType = dict[str, dict[str, dict[str, tuple[str, list[int]]]]]
SnomedMappingType = dict[str, dict[str, str]]


class _CID_Dict:
    repr_format = "{} = {}"
    str_format = "{:20} {:12} {:8} {}\n"

    def __init__(self, cid: int) -> None:
        self.cid = cid
        self._concepts: dict[str, Code] = {}

    def __dir__(self) -> list[str]:
        """Gives a list of available SR identifiers.

        List of attributes is used, for example, in auto-completion in editors
        or command-line environments.
        """
        meths = {v[0] for v in inspect.getmembers(type(self), inspect.isroutine)}
        props = {v[0] for v in inspect.getmembers(type(self), inspect.isdatadescriptor)}
        sr_names = set(self.dir())

        return sorted(props | meths | sr_names)

    def __getattr__(self, name: str) -> Code:
        """Return the ``Code`` for class attribute `name`."""
        matches = [
            scheme
            for scheme, keywords in CID_CONCEPTS[self.cid].items()
            if name in keywords
        ]

        if not matches:
            raise AttributeError(f"'{name}' not found in CID {self.cid}")

        if len(matches) > 1:
            # Should never happen, but just in case
            raise AttributeError(
                f"Multiple schemes found for '{name}' in CID {self.cid}: "
                f"{', '.join(matches)}"
            )

        scheme = matches[0]
        identifiers = cast(dict[str, tuple[str, list[int]]], CONCEPTS[scheme][name])
        # Almost always only one code per identifier
        if len(identifiers) == 1:
            code, val = list(identifiers.items())[0]
        else:
            _matches = [
                (code, val) for code, val in identifiers.items() if self.cid in val[1]
            ]
            if len(_matches) > 1:
                # Multiple codes shouldn't have the same keyword, but just in case
                codes = ", ".join([f"'{v[0]}'" for v in _matches])
                raise AttributeError(
                    f"'{name}' has multiple code matches in CID {self.cid}: {codes}"
                )

            code, val = _matches[0]

        return Code(value=code, meaning=val[0], scheme_designator=scheme)

    def __repr__(self) -> str:
        concepts = [
            self.repr_format.format(name, concept)
            for name, concept in self.concepts.items()
        ]

        return f"CID {self.cid}\n" + "\n".join(concepts)

    def __str__(self) -> str:
        """Return a str representation of the instance."""
        s = [f"CID {self.cid} ({name_for_cid[self.cid]})"]
        s.append(self.str_format.format("Attribute", "Code value", "Scheme", "Meaning"))
        s.append(self.str_format.format("---------", "----------", "------", "-------"))
        s.append(
            "\n".join(
                self.str_format.format(name, *concept)
                for name, concept in self.concepts.items()
            )
        )

        return "\n".join(s)


class Collection:
    """Interface for a collection of concepts, such as SNOMED-CT, or a DICOM CID.

    .. versionadded:: 3.0
    """

    repr_format = "{} = {}"

    def __init__(self, name: str) -> None:
        """Create a new collection.

        Parameters
        ----------
        name : str
            The name of the collection, should either be a key in the
            ``sr._concepts_dict.concepts`` :class:`dict` or a CID name for
            a CID in ``sr._cid_dict.cid_concepts`` such as ``"CID1234"``.
        """
        if not name.upper().startswith("CID"):
            self._name = name
            # dict[str, dict[str, tuple(str, list[int])]]
            # {'ACEInhibitor': {'41549009': ('ACE inhibitor', [3760])},
            self._scheme_data = CONCEPTS[name]
        else:
            self._name = f"CID{name[3:]}"
            # dict[str, list[str]]
            # {'SCT': ['Pericardium', 'Pleura', 'LeftPleura', 'RightPleura']}
            self._cid_data = CID_CONCEPTS[int(name[3:])]

        self._concepts: dict[str, Code] = {}

    @property
    def concepts(self) -> dict[str, Code]:
        """Return a :class:`dict` of {SR identifiers: codes}"""
        if not self._concepts:
            self._concepts = {name: getattr(self, name) for name in self.dir()}

        return self._concepts

    def __contains__(self, item: str | Code) -> bool:
        """Checks whether a given code is a member of the collection.

        Parameters
        ----------
        item : pydicom.sr.coding.Code | str
            The code to check for as either the code or the corresponding
            keyword.

        Returns
        -------
        bool
            Whether the collection contains the `code`
        """
        if isinstance(item, str):
            try:
                code = getattr(self, item)
            except AttributeError:
                return False
        else:
            code = item

        return code in self.concepts.values()

    def __dir__(self) -> list[str]:
        """Return a list of available concept keywords.

        List of attributes is used, for example, in auto-completion in editors
        or command-line environments.
        """
        meths = {v[0] for v in inspect.getmembers(type(self), inspect.isroutine)}
        props = {v[0] for v in inspect.getmembers(type(self), inspect.isdatadescriptor)}
        sr_names = set(self.dir())

        return sorted(props | meths | sr_names)

    def dir(self, *filters: str) -> list[str]:
        """Return an sorted list of SR identifiers based on a partial match.

        Parameters
        ----------
        filters : str
            Zero or more string arguments to the function. Used for
            case-insensitive match to any part of the SR keyword.

        Returns
        -------
        list of str
            The matching SR keywords. If no filters are used then all
            keywords are returned.
        """
        # CID_CONCEPTS: Dict[int, Dict[str, List[str]]]
        if self.is_cid:
            return _filtered(chain.from_iterable(self._cid_data.values()), filters)

        return _filtered(self._scheme_data, filters)

    def __getattr__(self, name: str) -> Code:
        """Return the :class:`~pydicom.sr.Code` corresponding to `name`.

        Parameters
        ----------
        name : str
            A camel case version of the concept's code meaning, such as
            ``"FontanelOfSkull" in the SCT coding scheme.

        Returns
        -------
        pydicom.sr.Code
            The :class:`~pydicom.sr.Code` corresponding to `name`.
        """
        if self.name.startswith("CID"):
            # Try DICOM's CID collections
            matches = [
                scheme
                for scheme, keywords in self._cid_data.items()
                if name in keywords
            ]
            if not matches:
                raise AttributeError(
                    f"No matching code for keyword '{name}' in {self.name}"
                )

            if len(matches) > 1:
                # Shouldn't happen, but just in case
                raise RuntimeError(
                    f"Multiple schemes found to contain the keyword '{name}' in "
                    f"{self.name}: {', '.join(matches)}"
                )

            scheme = matches[0]
            identifiers = cast(CIDValueType, CONCEPTS[scheme][name])

            if len(identifiers) == 1:
                code, val = list(identifiers.items())[0]
            else:
                cid = int(self.name[3:])
                _matches = [
                    (code, val) for code, val in identifiers.items() if cid in val[1]
                ]
                if len(_matches) > 1:
                    # Shouldn't happen, but just in case
                    codes = ", ".join(v[0] for v in _matches)
                    raise RuntimeError(
                        f"Multiple codes found for keyword '{name}' in {self.name}: "
                        f"{codes}"
                    )

                code, val = _matches[0]

            return Code(value=code, meaning=val[0], scheme_designator=scheme)

        # Try concept collections such as SCT, DCM, etc
        try:
            entries = cast(CIDValueType, self._scheme_data[name])
        except KeyError:
            raise AttributeError(
                f"No matching code for keyword '{name}' in scheme '{self.name}'"
            )

        if len(entries) > 1:
            # val is {"code": ("meaning", [cid1, cid2, ...], "code": ...}
            code_values = ", ".join(entries.keys())
            raise RuntimeError(
                f"Multiple codes found for keyword '{name}' in scheme '{self.name}': "
                f"{code_values}"
            )

        code = list(entries.keys())[0]  # get first and only
        meaning, cids = entries[code]

        return Code(value=code, meaning=meaning, scheme_designator=self.name)

    @property
    def is_cid(self) -> bool:
        """Return ``True`` if the collection is one of the DICOM CIDs"""
        return self.name.startswith("CID")

    @property
    def name(self) -> str:
        """Return the name of the collection."""
        return self._name

    def __repr__(self) -> str:
        """Return a representation of the collection."""
        concepts = [
            self.repr_format.format(name, concept)
            for name, concept in self.concepts.items()
        ]

        return f"{self.name}\n" + "\n".join(concepts)

    @property
    def scheme_designator(self) -> str:
        """Return the scheme designator for the collection."""
        return self.name

    def __str__(self) -> str:
        """Return a string representation of the collection."""
        len_names = max(len(n) for n in self.concepts.keys()) + 2
        len_codes = max(len(c[0]) for c in self.concepts.values()) + 2
        len_schemes = max(len(c[1]) for c in self.concepts.values()) + 2

        # Ensure each column is at least X characters wide
        len_names = max(len_names, 11)
        len_codes = max(len_codes, 6)
        len_schemes = max(len_schemes, 8)

        if self.is_cid:
            fmt = f"{{:{len_names}}} {{:{len_codes}}} {{:{len_schemes}}} {{}}"

            s = [self.name]
            s.append(fmt.format("Attribute", "Code", "Scheme", "Meaning"))
            s.append(fmt.format("---------", "----", "------", "-------"))
            s.append(
                "\n".join(
                    fmt.format(name, *concept)
                    for name, concept in self.concepts.items()
                )
            )
        else:
            fmt = f"{{:{len_names}}} {{:{len_codes}}} {{}}"

            s = [f"Scheme: {self.name}"]
            s.append(fmt.format("Attribute", "Code", "Meaning"))
            s.append(fmt.format("---------", "----", "-------"))

            s.append(
                "\n".join(
                    fmt.format(name, concept[0], concept[2])
                    for name, concept in self.concepts.items()
                )
            )

        return "\n".join(s)

    def trait_names(self) -> list[str]:
        """Return a list of valid names for auto-completion code.

        Used in IPython, so that data element names can be found and offered
        for autocompletion on the IPython command line.
        """
        return dir(self)


class _CodesDict:
    """Interface for a concepts dictionary.

    Examples
    --------
    >>> from pydicom.sr import codes
    >>> code = codes.SCT.Deep
    >>> code.value
    '795002'
    >>> code.meaning
    'Deep'
    >>> code == codes.CID2.Deep  # Or use the CID instead
    True
    >>> code = codes.SCT.FontanelOfSkull
    >>> code.value
    '79361005'
    >>> code.meaning
    'Fontanel of skull'
    """
    def __init__(self, scheme: str | None = None) -> None:
        """Create a new CodesDict.

        Parameters
        ----------
        scheme : str, optional
            The if used, then the scheme designator for the concepts
            dictionary.
        """
        self.scheme = scheme
        self._dict = {scheme: CONCEPTS[scheme]} if scheme else CONCEPTS

    def __dir__(self) -> list[str]:
        """Gives a list of available SR identifiers.

        List of attributes is used, for example, in auto-completion in editors
        or command-line environments.
        """
        meths = {v[0] for v in inspect.getmembers(type(self), inspect.isroutine)}
        props = {v[0] for v in inspect.getmembers(type(self), inspect.isdatadescriptor)}
        sr_names = set(self.dir())

        return sorted(props | meths | sr_names)

    def __getattr__(self, name: str) -> Union["_CodesDict", _CID_Dict, Code]:
        """Return either a ``_CodesDict``, ``_CID_Dict`` or ``Code`` depending
        on the `name`.

        Parameters
        ----------
        name : str
            One of the following:
                * A coding scheme designator such as ``"SCT"``.
                * A concept ID such as ``"CID2"``.
                * If ``_CodesDict.scheme`` is not ``None``, a camel case version
                  of the concept's code meaning, such as ``"FontanelOfSkull" in
                  the SCT coding scheme.

        Returns
        -------
        pydicom.sr._CodesDict, pydicom.sr._CID_Dict or pydicom.sr.Code

            * If `name` is a concept ID then the ``_CID_Dict`` for the
              corresponding CID.
            * If `name` is a coding scheme designator then the ``_CodesDict``
              instance for the corresponding scheme.
            * If ``_CodesDict.scheme`` is not ``None`` then the ``Code``
              corresponding to `name`.
        """
        # for codes.X, X must be a CID or a scheme designator
        if name.startswith("cid"):
            if not self.scheme:
                return _CID_Dict(int(name[3:]))
            raise AttributeError("Cannot use a CID with a scheme dictionary")

        if name in self._dict.keys():
            # Return concepts limited only the specified scheme designator
            return _CodesDict(scheme=name)

        # If not already narrowed to a particular scheme, is an error
        if not self.scheme:
            raise AttributeError(
                f"'{name}' not recognized as a CID or scheme designator"
            )

        # else try to find in this scheme
        try:
            val = cast(dict[str, tuple[str, list[int]]], self._dict[self.scheme][name])
        except KeyError:
            raise AttributeError(
                f"Unknown code name '{name}' for scheme '{self.scheme}'"
            )

        if len(val) > 1:
            # val is {code value: (meaning, cid_list}, code_value: ...}
            code_values = ", ".join(val.keys())
            raise RuntimeError(
                f"Multiple code values for '{name}' found: {code_values}"
            )

        code = list(val.keys())[0]  # get first and only
        meaning, cids = val[code]

        return Code(value=code, meaning=meaning, scheme_designator=self.scheme)

    def dir(self, *filters: str) -> list[str]:
        """Returns an alphabetical list of SR identifiers based on a partial match.

        Intended mainly for use in interactive Python sessions.
        """
        return _filtered(chain.from_iterable(self._dict.values()), filters)

    def schemes(self) -> KeysView[str]:
        return self._dict.keys()

    def trait_names(self) -> list[str]:
        """Returns a list of valid names for auto-completion code.

        Used in IPython, so that data element names can be found and offered
        for autocompletion on the IPython command line.
        """
        return dir(self)


# Named concept collections like SNOMED-CT, etc
_collections = [Collection(designator) for designator in CONCEPTS]
# DICOM CIDs
_collections.extend(Collection(f"CID{cid}") for cid in name_for_cid)

codes = _CodesDict()