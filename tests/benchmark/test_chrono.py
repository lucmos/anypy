from time import sleep

import pytest

from anypy.benchmark.chrono import Chrono, ChronoAlreadyStartedError, ChronoAlreadyStoppedError, ChronoRuntimeError


def test_instantiation_chrono_error():
    with pytest.raises(ChronoRuntimeError):
        Chrono(0, "test", 5)


def test_chrono_instatiation():
    with Chrono.start("test"):
        sleep(0.05)
        with Chrono.start("test2"):
            sleep(0.05)
        with Chrono.start("test3"):
            sleep(0.05)
            with Chrono.start("test5"):
                sleep(0.05)
        with Chrono.start("test4"):
            sleep(0.05)

    c = Chrono.start("test")
    sleep(0.05)
    c.stop()

    c = Chrono.start("test")
    sleep(0.05)
    c.elapsed()
    c.elapsed()
    c.elapsed()
    c.stop()


def test_chrono_already_stopped():
    c = Chrono.start("test")
    sleep(0.05)
    c.stop()
    with pytest.raises(ChronoAlreadyStoppedError):
        c.stop()

    with pytest.raises(ChronoAlreadyStoppedError):
        with (c := Chrono.start("test")):
            sleep(0.05)
            c.stop()

    with pytest.raises(ChronoAlreadyStoppedError):
        c.elapsed()

    with pytest.raises(ChronoAlreadyStoppedError):
        with (c := Chrono.start("test")):
            sleep(0.05)
        c.elapsed()


def test_chrono_already_started():
    c = Chrono.start("test")
    sleep(0.05)
    with pytest.raises(ChronoAlreadyStartedError):
        c.start("test")

    with pytest.raises(ChronoAlreadyStartedError):
        with (c := Chrono.start("test")):
            sleep(0.05)
            c.start("test")
