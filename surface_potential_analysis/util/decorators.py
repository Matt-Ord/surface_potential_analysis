from __future__ import annotations

import datetime
from collections.abc import Callable, Mapping
from functools import update_wrapper, wraps
from typing import TYPE_CHECKING, Any, Generic, ParamSpec, TypeVar, overload

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

_P = ParamSpec("_P")
_R = TypeVar("_R")
_RD = TypeVar("_RD", bound=Mapping[Any, Any])


def timed(f: Callable[_P, _R]) -> Callable[_P, _R]:
    """
    Log the time taken for f to run.

    Parameters
    ----------
    f : Callable[P, R]
        The function to time

    Returns
    -------
    Callable[P, R]
        The decorated function
    """

    @wraps(f)
    def wrap(*args: _P.args, **kw: _P.kwargs) -> _R:
        ts = datetime.datetime.now(tz=datetime.UTC)
        try:
            result = f(*args, **kw)
        finally:
            te = datetime.datetime.now(tz=datetime.UTC)
            print(f"func: {f.__name__} took: {(te - ts).total_seconds()} sec")  # noqa: T201
        return result

    return wrap  # type: ignore[return-value]


@overload
def npy_cached(
    path: Path | None,
    *,
    load_pickle: bool = False,
    save_pickle: bool = True,
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    ...


@overload
def npy_cached(
    path: Callable[_P, Path | None],
    *,
    load_pickle: bool = False,
    save_pickle: bool = True,
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    ...


def npy_cached(
    path: Path | None | Callable[_P, Path | None],
    *,
    load_pickle: bool = False,
    save_pickle: bool = True,
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """
    Cache the response of the function at the given path.

    Parameters
    ----------
    path : Path | Callable[P, Path]
        The file to read.
    load_pickle : bool, optional
        Allow loading pickled object arrays stored in npy files.
        Reasons for disallowing pickles include security, as loading pickled data can execute arbitrary code.
        If pickles are disallowed, loading object arrays will fail. default: False
    save_pickle : bool, optional
        Allow saving pickled objects. default: True

    Returns
    -------
    Callable[[Callable[P, R]], Callable[P, R]]
    """

    def _npy_cached(f: Callable[_P, _R]) -> Callable[_P, _R]:
        @wraps(f)
        def wrap(*args: _P.args, **kw: _P.kwargs) -> _R:
            cache_path = path(*args, **kw) if callable(path) else path
            if cache_path is None:
                return f(*args, **kw)
            try:
                arr: _R = np.load(cache_path, allow_pickle=load_pickle)[()]
            except FileNotFoundError:
                arr = f(*args, **kw)
                # Saving pickeld
                np.save(cache_path, np.asanyarray(arr), allow_pickle=save_pickle)

            return arr

        return wrap  # type: ignore[return-value]

    return _npy_cached


class NPYCachedFunction(Generic[_P, _RD]):
    """A function wrapper which is used to cache the output."""

    def __init__(
        self,
        function: Callable[_P, _RD],
        path: Path | None | Callable[_P, Path | None],
        *,
        load_pickle: bool = False,
        save_pickle: bool = True,
    ) -> None:
        self._inner = function
        self._path = path

        self.load_pickle = load_pickle
        self.save_pickle = save_pickle

    def __call__(self, *args: _P.args, **kw: _P.kwargs) -> _RD:
        """Call the function using the cache."""
        cache_path = self._path(*args, **kw) if callable(self._path) else self._path
        if cache_path is None:
            return self.call_uncached(*args, **kw)
        cache_path = cache_path.with_suffix(".npz")
        try:
            data = np.load(cache_path, allow_pickle=self.load_pickle)
            obj: _RD = {f: data[f][()] for f in data.files}  # type: ignore not _RD
        except FileNotFoundError:
            obj = self.call_uncached(*args, **kw)
            np.savez(cache_path, **obj)

        return obj

    def call_uncached(self, *args: _P.args, **kw: _P.kwargs) -> _RD:
        """Call the function, without using the cache."""
        return self._inner(*args, **kw)


@overload
def npy_cached_dict(
    path: Path | None,
    *,
    load_pickle: bool = False,
) -> Callable[[Callable[_P, _RD]], NPYCachedFunction[_P, _RD]]:
    ...


@overload
def npy_cached_dict(
    path: Callable[_P, Path | None],
    *,
    load_pickle: bool = False,
) -> Callable[[Callable[_P, _RD]], NPYCachedFunction[_P, _RD]]:
    ...


def npy_cached_dict(
    path: Path | None | Callable[_P, Path | None],
    *,
    load_pickle: bool = False,
) -> Callable[[Callable[_P, _RD]], NPYCachedFunction[_P, _RD]]:
    """
    Cache the response of the function at the given path.

    Parameters
    ----------
    path : Path | Callable[P, Path]
        The file to read.
    load_pickle : bool, optional
        Allow loading pickled object arrays stored in npy files.
        Reasons for disallowing pickles include security, as loading pickled data can execute arbitrary code.
        If pickles are disallowed, loading object arrays will fail. default: False
    save_pickle : bool, optional
        Allow saving pickled objects. default: True

    Returns
    -------
    Callable[[Callable[P, R]], Callable[P, R]]
    """

    def _npy_cached(f: Callable[_P, _RD]) -> NPYCachedFunction[_P, _RD]:
        return update_wrapper(NPYCachedFunction(f, path, load_pickle=load_pickle), f)  # type: ignore aaa

    return _npy_cached
