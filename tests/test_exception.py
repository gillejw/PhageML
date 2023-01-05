import pytest
from phageml.exceptions import PhageMLException

class TestException():
    '''Tests Exception Handling in PhageML'''
    def test_phagemlexception_throws_exception(self):
        '''Tests that the PhageMLException class is called when an error is raised'''
        with pytest.raises(PhageMLException) as err:
            raise PhageMLException(err)
