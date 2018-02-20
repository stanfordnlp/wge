#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os, shutil, argparse, re, json, traceback
from codecs import open
from ConfigParser import RawConfigParser
from collections import defaultdict

import boto
from boto.mturk.connection import MTurkRequestError
from boto.mturk.price import Price
from boto.mturk.qualification import (Qualifications, Requirement,
                                      LocaleRequirement, AdultRequirement,
                                      NumberHitsApprovedRequirement,
                                      PercentAssignmentsAbandonedRequirement,
                                      PercentAssignmentsApprovedRequirement,
                                      PercentAssignmentsRejectedRequirement,
                                      PercentAssignmentsReturnedRequirement,
                                      PercentAssignmentsSubmittedRequirement)
from boto.mturk.question import ExternalQuestion

# 1 Batch has many HITs; 1 HIT has many assignments
# The Worker accepts an assignment, then either submits or returns it.
# The Requester reviews an assignment

# There are 5 HIT states:
# - Assignable:
#     The HIT is not expired *AND* at least 1 assignment has not been accepted
#     I.e., a Worker can accept an assignment from this HIT.
#     Extending a HIT will change the HIT to Assignable state.
# - Unassignable: <-- This name is the worst name ever
#     No assignment is in open state,
#     and at least one assignment is in assigned state.
#     I.e., no Workers can accept an assignment from this HIT.
# - Reviewable:
#     *ALL* assignments of the HIT are submitted *OR* the HIT expired.
#     The Requester can get the list of all reviewable HITs by calling
#     GetReviewableHITs.
#     Note that even though all assignments are approved or rejected, the HIT
#     will still be in Reviewable state. The Requester must call DisposeHIT
#     to dispose the HIT.
# - Reviewing (optional):
#     The Requester manually called SetHITAsReviewing on the HIT.
#     Benefit: GetReviewableHITs does not return HITs in Reviewing state.
# - Disposed:
#     The HIT has been deleted and can no longer be retrieved.
#     (HITs are automatically disposed after 120 days.)

# There are 6 assignment states:
# - Open (a Worker can accept the assignment)
# - Expired
# - Assigned (a Worker is working on the assignment)
# - Submitted (a Worker has submitted the assignment)
# - Approved
# - Rejected

class MTurkWrapper(object):
    SANDBOX = 'mechanicalturk.sandbox.amazonaws.com'
    PREVIEW_REAL = 'https://www.mturk.com/mturk/preview?groupId='
    PREVIEW_SANDBOX = 'https://workersandbox.mturk.com/mturk/preview?groupId='

    def __init__(self, sandbox=True):
        self.sandbox = sandbox

    def load(self):
        if self.sandbox:
            print "Using SANDBOX ..."
            self.mtc = boto.connect_mturk(host=MTurkWrapper.SANDBOX)
        else:
            print "Using REAL MTURK!"
            self.mtc = boto.connect_mturk()

    def get_account_balance(self):
        return self.mtc.get_account_balance()

    ################ CREATE HIT ################

    def _replace_variables(self, s, variables):
        for key, value in variables.iteritems():
            s = re.sub(r'\$\{' + key + r'\}', str(value), s, flags=re.I)
        assert '${' not in s, s
        return s

    def create_batch(self, properties, variables_list, maxhits=None):
        '''Create a new batch of HITs.
        Return (list of hit_ids, hit_type_id).
        'variables_list' is a list of dicts {key1: value1, key2: value2, ...}
        Strings of the form '${key1}' in question and annotation
        will be replaced with value1 and so on.
        '''
        if maxhits:
            variables_list = variables_list[:maxhits]
        if raw_input('Creating %d HITs. Continue? (y/N) '
                     % len(variables_list)).lower() != 'y':
            return
        # Register the HIT type
        if 'hittypeid' in properties:
            hit_type_id = properties['hittypeid']
        else:
            result = self.mtc.register_hit_type(
                properties['title'], properties['description'],
                properties['reward'], properties['assignmentduration'],
                properties['keywords'], properties['autoapprovaldelay'],
                properties['qualifications'])
            hit_type_id = result[0].HITTypeId
        # Reading parameters for individual HITs
        hit_ids = []
        for i, variables in enumerate(variables_list):
            question = ExternalQuestion(
                self._replace_variables(properties['url'], variables),
                properties['frameheight'])
            annotation = self._replace_variables(
                properties['annotation'], variables)
            if isinstance(properties['assignments'], int):
                max_assignments = properties['assignments']
            else:
                max_assignments = properties['assignments'][i]
            if max_assignments <= 0:
                print '(%5d/%5d)' % (i + 1, len(variables_list)),
                print 'Skipped because assignments <= 0'
                continue
            result = self.mtc.create_hit(
                hit_type=hit_type_id, question=question,
                annotation=annotation, lifetime=properties['hitlifetime'],
                max_assignments=max_assignments)
            hit_id = result[0].HITId
            hit_ids.append(hit_id)
            assert hit_type_id == result[0].HITTypeId
            print '(%5d/%5d)' % (i + 1, len(variables_list)),
            print 'Created HIT', hit_id
        print ('DONE! %d HITs created. Preview the HITs here:'
               % len(variables_list))
        if self.sandbox:
            print MTurkWrapper.PREVIEW_SANDBOX + hit_type_id
        else:
            print MTurkWrapper.PREVIEW_REAL + hit_type_id
        return hit_ids, hit_type_id
        
    def extend_batch(self, hit_ids, assignments_increment=None,
                     expiration_increment=None):
        '''Extend a batch of HITs.'''
        print 'Extending batch ...'
        print 'Assignment +=', assignments_increment
        print 'Expiration +=', expiration_increment
        for i, hit_id in enumerate(hit_ids):
            self.mtc.extend_hit(hit_id,
                                assignments_increment=assignments_increment,
                                expiration_increment=expiration_increment)
            print '(%5d/%5d)' % (i + 1, len(hit_ids)),
            print 'Extended', hit_id
        print 'Done!'

    ################ GET RESULTS ################

    def get_batch(self, hit_ids, status=None):
        '''Return a list of SUBMITTED assignments in the batch.
        Parameter 'status' can be one of
        - None (everything)
        - 'Submitted' (neither approved nor rejected yet)
        - 'Approved'
        - 'Rejected'
        - 'Approved,Rejected' (either approved or rejected)
        '''
        print 'Getting submitted assignments ...'
        assignments = []
        total_max_assignments = 0
        for i, hit_id in enumerate(hit_ids):
            result_set = self.mtc.get_assignments(hit_id, status, page_size=100)
            hit = self.mtc.get_hit(hit_id)[0]
            max_assignments = int(hit.MaxAssignments)
            total_max_assignments += max_assignments
            print '(%5d/%5d)' % (i + 1, len(hit_ids)),
            print hit_id, ':', result_set.NumResults, '/', max_assignments, 'assignments'
            assignments.extend(result_set)
        print 'DONE! %d / %d assignments retrieved.' % (len(assignments), total_max_assignments)
        return assignments

    ################ APPROVE / REJECT ################

    def _read_mapping(self, mapping):
        '''Return a list of (id, reason)
        mapping can be one of the following:
        - list or tuple of ids (reason = None)
        - dict from string (id) to string (reason)
        - dict from string (reason) to list or tuple (ids)
        '''
        if isinstance(mapping, (list, tuple)):
            return [(x, None) for x in mapping]
        elif isinstance(mapping, dict):
            items = mapping.items()
            if isinstance(items[0][1], (list, tuple)):
                return [(x, reason) for (reason, ids) in items for x in ids]
            else:
                return items
        assert False, 'mapping has incorrect type %s' % type(mapping)

    def approve_assignments(self, mapping):
        mapping = self._read_mapping(mapping)
        if raw_input('Approving %d assignments. Continue? (y/N) '
                     % len(mapping)).lower() != 'y':
            return
        for assignment_id, reason in mapping:
            try:
                self.mtc.approve_assignment(assignment_id, reason)
                print 'Approved %s (%s)' % (assignment_id, reason)
            except Exception, e:
                print e

    def reject_assignments(self, mapping):
        mapping = self._read_mapping(mapping)
        if raw_input('Rejecting %d assignments. Continue? (y/N) '
                     % len(mapping)).lower() != 'y':
            return
        for assignment_id, reason in mapping:
            self.mtc.reject_assignment(assignment_id, reason)
            print 'Rejected %s (%s)' % (assignment_id, reason)

    def approve_rejected_assignments(self, mapping):
        mapping = self._read_mapping(mapping)
        if raw_input('Resurrecting %d assignments. Continue? (y/N) '
                     % len(mapping)).lower() != 'y':
            return
        for assignment_id, reason in mapping:
            self.mtc.approve_rejected_assignment(assignment_id, reason)
            print 'Resurrected %s (%s)' % (assignment_id, reason)

    def grant_bonus(self, data):
        '''data = list of (worker_id, assignment_id, bonus_amount, reason)'''
        if raw_input('Granting bonus to %d Turkers. Continue? (y/N) '
                     % len(data)).lower() != 'y':
            return
        for worker_id, assignment_id, bonus_amount, reason in data:
            bonus_amount = Price(float(bonus_amount))
            self.mtc.grant_bonus(worker_id, assignment_id, bonus_amount, reason)
            print 'Granted %s to %s (%s)' % (bonus_amount, worker_id, reason)

    def block_workers(self, mapping):
        mapping = self._read_mapping(mapping)
        pass

    def unblock_workers(self, mapping):
        mapping = self._read_mapping(mapping)
        pass

    ################ CLEAN UP ################

    def delete_batch(self, hit_ids):
        '''Delete the HITs:
        - Try to dispose the HIT.
        - If failed (because the conditions of dispose_hit are not met),
          expire the HIT, approve the remaining assignments, and 
          re-dispose the HIT.
        '''
        if raw_input('Deleting %d HITs. Continue? (y/N) '
                     % len(hit_ids)).lower() != 'y':
            return
        for i, hit_id in enumerate(hit_ids):
            status = self.mtc.get_hit(hit_id)[0].HITStatus
            if status == 'Disposed':
                print '(%5d/%5d)' % (i + 1, len(hit_ids)),
                print 'HIT', hit_id, 'already disposed.'
                continue
            try:
                self.mtc.dispose_hit(hit_id)
                print '(%5d/%5d)' % (i + 1, len(hit_ids)),
                print 'Disposed HIT', hit_id
            except MTurkRequestError, e:
                print 'Trying to dispose HIT', hit_id, '...'
                try:
                    self.mtc.expire_hit(hit_id)
                    result_set = self.mtc.get_assignments(
                        hit_id, 'Submitted', page_size=100)
                    if len(result_set) > 0:
                        print 'Approving %d assignments ...' % len(result_set)
                    for assignment in result_set:
                        self.mtc.approve_assignment(assignment.AssignmentId)
                    self.mtc.dispose_hit(hit_id)
                    print '(%5d/%5d)' % (i + 1, len(hit_ids)),
                    print 'Disposed HIT', hit_id
                except MTurkRequestError, e:
                    traceback.print_exc()
                    exit(1)
        print 'DONE! %d HITs disposed.' % len(hit_ids)

    def early_expire_hits(self, hit_ids):
        '''Expire several HITs'''
        if raw_input('Expiring %d HITs. Continue? (y/N) '
                     % len(hit_ids)).lower() != 'y':
            return
        for i, hit_id in enumerate(hit_ids):
            self.mtc.expire_hit(hit_id)
            print '(%5d/%5d)' % (i + 1, len(hit_ids)),
            print 'Expired HIT', hit_id
        print 'DONE! %d HITs expired.' % len(hit_ids)

    def dispose_batch(self, hit_ids):
        '''Dispose HITs such that
        - the HIT is in REVIEWABLE state, and
        - all assignments approved or rejected.
        If not all conditions are met, an error is thrown.
        Warning: After disposing the HIT, the Requester can no longer approve
        the rejected assignments.
        The results can still be downloaded until 120 days after.
        '''
        pass

    def disable_hit(self, hit_ids):
        '''Deal with HITs that are NOT REVIEWABLE:
        - Remove HITs from marketplace
        - Approve all submitted assignments (+ Pay workers)
          (that haven't been accepted or rejected),
        - Dispose of the HITs and all assignment data.
        Assignment results data CANNOT be retreived in the future!
        '''
        pass

    ################ EMERGENCY ################

    def get_all_hits(self):
        '''Return the list of all (HIT id, HIT type id)'''
        for x in self.mtc.get_all_hits():
            print '%s\t%s' % (x.HITId, x.HITTypeId)

################################################################

class RecordWrapper(object):
    def __init__(self, basedir):
        assert os.path.isdir(basedir)
        self.basedir = basedir
        self.dirname = os.path.basename(os.path.realpath(basedir))

    def _get_filename(self, extension, check=False):
        filename = os.path.join(self.basedir, self.dirname + '.' + extension)
        if check and os.path.exists(filename):
            confirm = raw_input('%s exists. Overwrite? (Yes/No/Rename) ' % filename)
            if confirm.lower() == 'r':
                suffix = 0
                while os.path.exists(filename + '.conflict.' + str(suffix)):
                    suffix += 1
                return filename + '.conflict.' + str(suffix)
            if confirm.lower() != 'y':
                return None
        return filename

    TIME_MULTIPLIERS = {'s': 1, 'm': 60, 'h': 60 * 60, 'd': 60 * 60 * 24,
                        'w': 60 * 60 * 24 * 7}
    def _parse_time(self, timespec):
        if timespec[-1] in RecordWrapper.TIME_MULTIPLIERS:
            return int(float(timespec[:-1]) * 
                       RecordWrapper.TIME_MULTIPLIERS[timespec[-1]])
        return int(timespec)

    QUALIFICATIONS = {'adult': AdultRequirement,
                      'numapproved': NumberHitsApprovedRequirement,
                      '%abandoned': PercentAssignmentsAbandonedRequirement,
                      '%approved': PercentAssignmentsApprovedRequirement,
                      '%rejected': PercentAssignmentsRejectedRequirement,
                      '%returned': PercentAssignmentsReturnedRequirement,
                      '%submitted': PercentAssignmentsSubmittedRequirement}
    COMPARATORS = {'<': 'LessThan', '<=': 'LessThanOrEqualTo',
                   '>': 'GreaterThan', '>=': 'GreaterThanOrEqualTo',
                   '=': 'EqualTo', '!=': 'NotEqualTo'}

    def read_config(self):
        '''Return (properties, variables_list)'''
        filename = self._get_filename('config')
        parser = RawConfigParser()
        parser.read(filename)
        properties = {}
        if parser.has_option('properties', 'hittypeid'):
            properties['hittypeid'] = parser.get('properties', 'hittypeid')
        else:
            # Create a new HIT Type ID if not present
            for key in ('title', 'description', 'keywords'):
                properties[key] = parser.get('properties', key)
            properties['reward'] = Price(parser.getfloat('properties', 'reward'))
            for key in ('assignmentduration', 'autoapprovaldelay'):
                properties[key] = self._parse_time(parser.get('timing', key))
            # Qualifications
            requirements = []
            if parser.has_option('qualifications', 'locale'):
                requirements.append(LocaleRequirement(
                        'EqualTo', parser.get('qualifications', 'locale'), True))
            for key in RecordWrapper.QUALIFICATIONS:
                if parser.has_option('qualifications', key):
                    value = parser.get('qualifications', key)
                    comparator = ''.join(x for x in value if not x.isdigit())
                    value = int(value[len(comparator):])
                    requirements.append(RecordWrapper.QUALIFICATIONS[key](
                            RecordWrapper.COMPARATORS[comparator], value, True))
            properties['qualifications'] = Qualifications(requirements)
        # Other properties
        properties['annotation'] = parser.get('properties', 'annotation')
        properties['assignments'] = parser.get('properties', 'assignments')
        try:
            properties['assignments'] = int(properties['assignments'])
        except ValueError:
            properties['assignments'] = self.read_assignment_amounts(properties['assignments'])
        properties['hitlifetime'] = self._parse_time(parser.get('timing', 'hitlifetime'))
        # Question
        properties['url'] = parser.get('question', 'url')
        properties['frameheight'] = parser.get('question', 'frameheight')
        # Input
        n = parser.getint('input', 'numhits')
        if isinstance(properties['assignments'], list):
            assert len(properties['assignments']) == n, (len(properties['assignments']), n)
        variables_list = [dict() for i in xrange(n)]
        for key in parser.options('input'):
            if key != 'numhits':
                value = parser.get('input', key)
                if value[0] == '[':
                    value = json.loads(value)
                    assert len(value) == n
                    for i in xrange(n):
                        variables_list[i][key] = value[i]
                elif '-' in value:
                    start, end = [int(x) for x in value.split('-')]
                    assert end - start + 1 == n
                    for i in xrange(n):
                        variables_list[i][key] = start + i
                else:
                    for i in xrange(n):
                        variables_list[i][key] = value
        return properties, variables_list

    def read_assignment_amounts(self, suffix):
        filename = self._get_filename(suffix)
        with open(filename, 'r', 'utf8') as fin:
            return [int(x) for x in fin if x.strip()]

    def read_increments(self):
        '''Return (assignments_increment, expiration_increment)'''
        a_i = raw_input('Assignment increment: ')
        try:
            a_i = int(a_i) or None
        except:
            print 'Invalid input "%s". Set to None.' % a_i
            a_i = None
        e_i = raw_input('Expiration increment: ')
        try:
            e_i = self._parse_time(e_i) or None
        except:
            print 'Invalid input "%s". Set to None.' % e_i
            e_i = None
        print '>>> Assignment +=', a_i
        print '>>> Expiration +=', e_i
        if raw_input('Is this OK? (Yes/No) ').lower()[:1] == 'y':
            return (a_i, e_i)
        return self.read_increments()

    def write_success(self, hit_ids, hit_type_id):
        filename = self._get_filename('success', check=True)
        if not filename:
            return
        with open(filename, 'w', 'utf8') as fout:
            print >> fout, '\t'.join(('hitId', 'hitTypeId'))
            for hit_id in hit_ids:
                print >> fout, '\t'.join((hit_id, hit_type_id))

    def read_success(self):
        '''Return HIT IDs'''
        with open(self._get_filename('success')) as fin:
            return [line.split()[0] for line in fin.readlines()[1:]]

    def read_expire(self):
        '''Return HIT IDs'''
        with open(self._get_filename('expire')) as fin:
            return [line.split()[0] for line in fin.readlines()[1:]]

    ASSIGNMENT_FIELDS = (
        'AssignmentId', 'WorkerId', 'HITId',
        'AssignmentStatus', # Submitted / Approved / Rejected
        'AcceptTime', 'SubmitTime', 'AutoApprovalTime',
        'ApprovalTime', 'RejectionTime',
        )

    def write_results(self, assignments):
        filename = self._get_filename('results', check=False)
        if not filename:
            return
        records = []
        statistics = defaultdict(int)
        for assignment in assignments:
            statistics[assignment.AssignmentStatus] += 1
            record = {'metadata': {}, 'answers': {}}
            for key in RecordWrapper.ASSIGNMENT_FIELDS:
                try:
                    record['metadata'][key] = getattr(assignment, key)
                except AttributeError:
                    pass    # Ignore field
            for answer in assignment.answers[0]:
                record['answers'][answer.qid] = answer.fields[0]
            records.append(record)
        with open(filename, 'w', 'utf8') as fout:
            json.dump(records, fout, ensure_ascii=False, indent=2,
                      separators=(',', ': '), sort_keys=True)
        print ('Wrote %d records to %s' % (len(records), filename))
        for key, value in statistics.iteritems():
            print '%12s: %6d / %6d (%8.3f%%)' % (key, value, len(records),
                                                 value * 100.0 / len(records))

    def read_results(self):
        '''Return a list of {'metadata': {...}, 'answers': {...}}'''
        filename = self._get_filename('results')
        with open(filename, 'r', 'utf8') as fin:
            return json.load(fin)

    def _read_approve_or_reject(self, fin):
        '''Return a mapping from assignment_id to reason
        Format:
        # Reason for assignment IDs below <-- The first one is optional
        Assignment ID
        Assignment ID
        ...
        # Reason for worker IDs below
        Assignment ID
        ...
        '''
        mapping = {}
        reason = ''
        for line in fin:
            line = line.strip()
            if line.startswith('#'):
                reason = line[1:].strip()
            elif line:
                mapping[line] = reason
        return mapping

    def read_approve(self):
        filename = self._get_filename('approve')
        if not os.path.exists(filename):
            return None
        with open(filename, 'r', 'utf8') as fin:
            return self._read_approve_or_reject(fin)

    def read_reject(self):
        filename = self._get_filename('reject')
        if not os.path.exists(filename):
            return None
        with open(filename, 'r', 'utf8') as fin:
            return self._read_approve_or_reject(fin)

    def read_tsv(self, extension):
        """If all else fails..."""
        filename = self._get_filename(extension)
        if not os.path.exists(filename):
            return None
        with open(filename, 'r', 'utf8') as fin:
            return [x.strip().split('\t') for x in fin if x.strip()]

################################################################

class Actions(object):
    ACTIONS = ('getbalance', 'create', 'extend', 'get', 'clean',
               'grade', 'approve', 'reject', 'expire', 'bonus',
               'getallhits')

    def __init__(self, sandbox=True, basedir=None):
        self.mturk_w = MTurkWrapper(sandbox=sandbox)
        if basedir:
            self.record_w = RecordWrapper(basedir)
        else:
            self.record_w = None

    def getbalance(self, args):
        """ Print the balance and exit.
        
        Does not require any file, but you still need to specify a dummy directory
        in the command line.

        To get real MTurk balance, add the --real flag.
        """
        self.mturk_w.load()
        print self.mturk_w.get_account_balance()

    def create(self, args):
        """ Create a batch of HITs.

        Requires [name].config containing the HIT configurations.
        See the example config file.

        Creates [name].success containing created HIT IDs.

        Make sure you have enough balance first.
        Otherwise it is pretty difficult to fix the error.
        """
        properties, variables_list = self.record_w.read_config()
        print '=' * 40
        for key in sorted(properties):
            print key, ':', properties[key]
        print '=' * 40
        self.mturk_w.load()
        response = self.mturk_w.create_batch(
            properties, variables_list, maxhits=args.maxhits)
        if response:
            hit_ids, hit_type_id = response
            self.record_w.write_success(hit_ids, hit_type_id)

    def extend(self, args):
        """ Extend an existing batch of HITs.

        Requires [name].success containing HIT IDs (created by |create|).

        Creates a new [name].success file; the old file will be backed up.

        You will be prompted to enter the amount of time and assignments per HIT to add.
        Either fields can be left blank.
        Time = number of seconds, but you can use shorthands like 1d (= 1 day)
        """
        hit_ids = self.record_w.read_success()
        assignments_increment, expiration_increment =\
            self.record_w.read_increments()
        self.mturk_w.load()
        self.mturk_w.extend_batch(hit_ids,
                                  assignments_increment=assignments_increment,
                                  expiration_increment=expiration_increment)

    def get(self, args):
        """ Retrieve Turker's work for a batch of HITs.

        Requires [name].success containing HIT IDs (created by |create|).

        Creates [name].results, a JSON file containing the results.
        """
        hit_ids = self.record_w.read_success()
        self.mturk_w.load()
        assignments = self.mturk_w.get_batch(hit_ids)
        self.record_w.write_results(assignments)

    def clean(self, args):
        """ Remove a batch of HITs from Amazon permanently.

        Requires [name].success containing HIT IDs (created by |create|).

        You should only call |clean| on sandbox tasks.
        For the real tasks, just leave it on Amazon.
        """
        hit_ids = self.record_w.read_success()
        self.mturk_w.load()
        self.mturk_w.delete_batch(hit_ids)

    def grade(self, args):
        """ Perform |reject| and then |approve|. (Shortcut)

        Requires at least one of [name].approve and [name].reject
        See |approve| and |reject| for file description.

        After all assignments are approved or rejected, back up the [name].approve
        and [name].reject by renaming them as [name].approve-## and [name].reject-##
        (## = number).
        """
        mapping_rej = self.record_w.read_reject()
        mapping_app = self.record_w.read_approve()
        if not (mapping_rej or mapping_app):
            print 'Nothing to reject or approve.'
            exit(0)
        i = 1
        while os.path.exists(self.record_w._get_filename('approve-%02d' % i)) \
                or os.path.exists(self.record_w._get_filename('reject-%02d' % i)):
            i += 1
        print 'Reject, Approve, and move files to ...-%02d' % i
        self.mturk_w.load()
        if mapping_rej:
            self.mturk_w.reject_assignments(mapping_rej)
            shutil.move(self.record_w._get_filename('reject'),
                        self.record_w._get_filename('reject-%02d' % i))
        else:
            print 'No assignment to reject.'
        if mapping_app:
            self.mturk_w.approve_assignments(mapping_app)
            shutil.move(self.record_w._get_filename('approve'),
                        self.record_w._get_filename('approve-%02d' % i))
        else:
            print 'No assignment to approve.'

    def approve(self, args):
        """ Approve assignments from the given list.
        It is better to use |grade| since it also handles |reject| and backs up files.

        Requires [name].approve containing one assignment ID per line.

        To give a feedback message to the approved assignments, add a comment line
        in [name].approve (like "# Your answer is awesome."). All assignments
        after that comment line will have that message. Later comment lines
        override the previous ones.
        """
        mapping = self.record_w.read_approve()
        self.mturk_w.load()
        self.mturk_w.approve_assignments(mapping)

    def reject(self, args):
        """ Reject assignments from the given list.
        It is better to use |grade| since it also handles |approve| and backs up files.

        Requires [name].reject containing one assignment ID per line.

        To give a feedback message to the rejected assignments, add a comment line
        in [name].reject (like "# Your answer is nonsense."). All assignments
        after that comment line will have that message. Later comment lines
        override the previous ones.
        """
        mapping = self.record_w.read_reject()
        self.mturk_w.load()
        self.mturk_w.reject_assignments(mapping)

    def expire(self, args):
        """ Immediately expire a batch of HITs.

        Requires [name].success containing HIT IDs (created by |create|).
        """
        hit_ids = self.record_w.read_expire()
        self.mturk_w.load()
        self.mturk_w.early_expire_hits(hit_ids)

    def bonus(self, args):
        """ Give bonus to workers in a list.

        Requires [name].bonus containing one worker ID per line.

        To give a feedback message to the approved workers, add a comment line
        in [name].bonus (like "# Your work is awesome."). All workers
        after that comment line will have that message. Later comment lines
        override the previous ones.
        """
        data = self.record_w.read_tsv('bonus')
        self.mturk_w.load()
        self.mturk_w.grant_bonus(data)

    def getallhits(self, args):
        """ Get the list of all HITs ever published in the account.

        If something fails, use this as a last resort for debugging stuff.
        """
        self.mturk_w.load()
        self.mturk_w.get_all_hits()

################################################################

class CustomArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        print >> sys.stderr, 'ERROR:', message
        self.print_help()
        sys.exit(2)

if __name__ == '__main__':
    parser = CustomArgumentParser()
    parser.add_argument('--real', action='store_false', dest='sandbox', default=True,
                        help="Use the real MTurk instead of sandbox")
    parser.add_argument('--maxhits', type=int,
                        help='Maximum number of HITs (for debugging in sandbox)')
    parser.add_argument('dir',
                        help="Base directory")
    parser.add_argument('action',
                        help="action to take (%s)" % ', '.join(Actions.ACTIONS) +
                        " Read the Action class in run.py to see what each action does")
    args = parser.parse_args()

    # If action comes before dir (typo) ...
    if not os.path.exists(args.dir) and os.path.exists(args.action):
        args.dir, args.action = args.action, args.dir
    # Perform action
    actions = Actions(args.sandbox, args.dir)
    if hasattr(actions, args.action.lower()):
        getattr(actions, args.action.lower())(args)
    else:
        print "Action '%s' not recognized" % args.action
