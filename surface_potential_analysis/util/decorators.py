from __future__ import annotations

import datetime
import pickle  # noqa: S403
from collections.abc import Callable, Mapping
from functools import update_wrapper, wraps
from typing import TYPE_CHECKING, Any, Generic, Literal, ParamSpec, TypeVar, overload

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


CallType = Literal[
    "load_or_call_cached",
    "load_or_call_uncached",
    "call_uncached",
    "call_cached",
]


class CachedFunction(Generic[_P, _RD]):
    """A function wrapper which is used to cache the output."""

    def __init__(
        self,
        function: Callable[_P, _RD],
        path: Path | None | Callable[_P, Path | None],
        *,
        default_call: CallType = "load_or_call_cached",
    ) -> None:
        self._inner = function
        self._path = path

        self.default_call: CallType = default_call

    def _get_cache_path(self, *args: _P.args, **kw: _P.kwargs) -> Path | None:
        cache_path = self._path(*args, **kw) if callable(self._path) else self._path
        if cache_path is None:
            return None
        return cache_path

    def call_uncached(self, *args: _P.args, **kw: _P.kwargs) -> _RD:
        """Call the function, without using the cache."""
        return self._inner(*args, **kw)

    def call_cached(self, *args: _P.args, **kw: _P.kwargs) -> _RD:
        """Call the function, and save the result to the cache."""
        obj = self.call_uncached(*args, **kw)
        cache_path = self._get_cache_path(*args, **kw)
        if cache_path is not None:
            with cache_path.open("wb") as f:
                pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        return obj

    def _load_cache(self, *args: _P.args, **kw: _P.kwargs) -> _RD | None:
        """Call the function, delete and ."""
        cache_path = self._get_cache_path(*args, **kw)
        if cache_path is None:
            return None
        try:
            with cache_path.open("rb") as f:
                return pickle.load(f)  # noqa: S301
        except FileNotFoundError:
            return None

    def load_or_call_uncached(self, *args: _P.args, **kw: _P.kwargs) -> _RD:
        """Call the function uncached, using the cached data if available."""
        obj = self._load_cache(*args, **kw)

        if obj is None:
            obj = self.call_uncached(*args, **kw)
        return obj

    def load_or_call_cached(self, *args: _P.args, **kw: _P.kwargs) -> _RD:
        """Call the function cached, using the cached data if available."""
        obj = self._load_cache(*args, **kw)

        if obj is None:
            obj = self.call_cached(*args, **kw)
        return obj

        return obj

    def __call__(self, *args: _P.args, **kw: _P.kwargs) -> _RD:
        """Call the function using the cache."""
        match self.default_call:
            case "call_cached":
                return self.call_cached(*args, **kw)
            case "call_uncached":
                return self.call_uncached(*args, **kw)
            case "load_or_call_cached":
                return self.load_or_call_cached(*args, **kw)
            case "load_or_call_uncached":
                return self.load_or_call_uncached(*args, **kw)


@overload
def cached(
    path: Path | None,
    *,
    default_call: CallType = "load_or_call_cached",
) -> Callable[[Callable[_P, _RD]], CachedFunction[_P, _RD]]:
    ...


@overload
def cached(
    path: Callable[_P, Path | None],
    *,
    default_call: CallType = "load_or_call_cached",
) -> Callable[[Callable[_P, _RD]], CachedFunction[_P, _RD]]:
    ...


def cached(
    path: Path | None | Callable[_P, Path | None],
    *,
    default_call: CallType = "load_or_call_cached",
) -> Callable[[Callable[_P, _RD]], CachedFunction[_P, _RD]]:
    """
    Cache the response of the function at the given path using pickle.

    Parameters
    ----------
    path : Path | Callable[P, Path]
        The file to read.

    Returns
    -------
    Callable[[Callable[P, R]], Callable[P, R]]
    """

    def _cached(f: Callable[_P, _RD]) -> CachedFunction[_P, _RD]:
        return update_wrapper(  # type: ignore aaa
            CachedFunction(f, path, default_call=default_call),
            f,
        )

    return _cached
