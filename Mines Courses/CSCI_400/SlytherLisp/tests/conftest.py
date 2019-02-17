import pytest


deliverables = ['d1', 'd2', 'd3', 'd4']


def pytest_addoption(parser):
    for d in deliverables:
        parser.addoption(
            '--' + d,
            action='store_true',
            help='Run tests for {}'.format(d.upper()))


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    orig = items[:]
    items.clear()
    for item in orig:
        d = item.location[0].split('/')[-2]
        if ((d in deliverables and config.getoption(d))
                or d not in deliverables):
            items.append(item)
