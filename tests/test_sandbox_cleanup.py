"""Interpret Docker inspect evidence only; no Docker/GPU/model invocation."""
from types import SimpleNamespace
import unittest

from robotagentbench.model_examples.codex.sandbox_smoke import _inspect_confirms_removed


CID='a1'*32


def result(stderr,*,stdout='[]\n',returncode=1):
    return SimpleNamespace(returncode=returncode,stdout=stdout,stderr=stderr)


class SandboxCleanupTests(unittest.TestCase):
    def test_explicit_missing_same_id_all_supported_cli_case_forms(self):
        for prefix in ('','Error: ','error: ','Error response from daemon: '):
            for phrase in ('No such object: ','no such object: ','No such container: ','NO SUCH CONTAINER: '):
                for output in ('[]\n','', '  [ ]\n'):
                    with self.subTest(prefix=prefix,phrase=phrase,output=output):
                        self.assertTrue(_inspect_confirms_removed(result(prefix+phrase+CID+'\n',stdout=output),CID))

    def test_daemon_permission_transport_and_unrelated_errors_are_uncertain(self):
        for error in ('Cannot connect to the Docker daemon', 'permission denied',
                'context deadline exceeded', 'No such image: '+CID,
                'error: no such object: '+'b'*64,
                'error: no such object: '+CID+'-other',
                'error: no such object: '+CID+'\npermission denied',
                'warning: No such container: '+CID,
                'No such file or directory', 'permission denied (No such object: '+CID+')',
                'no such object: '+CID[:12], ''):
            with self.subTest(error=error):
                self.assertFalse(_inspect_confirms_removed(result(error),CID))

    def test_success_or_nonempty_inspect_output_never_counts_as_absence(self):
        error='error: no such object: '+CID
        for code in (0,None,True):
            self.assertFalse(_inspect_confirms_removed(result(error,returncode=code),CID))
        for output in ('[{"Id":"'+CID+'"}]','{}','null','garbage',None):
            self.assertFalse(_inspect_confirms_removed(result(error,stdout=output),CID))

    def test_missing_or_partial_container_identity_is_not_accepted(self):
        for identifier in ('',CID[:12],CID.upper(),'container-name',None):
            self.assertFalse(_inspect_confirms_removed(result('No such object: '+str(identifier)),identifier))


if __name__=='__main__':unittest.main()
