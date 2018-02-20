import random

from wge.miniwob.program import ExecutionEnvironment, ClickToken, \
        StringToken, UtteranceSelectorToken, InputElementsToken, \
        ButtonsToken, TextToken, LastToken, NearToken, LikeToken, \
        FocusAndTypeToken, FieldsValueSelectorToken, SameRowToken
from wge.miniwob.program_policy import LinearProgramPolicy
from wge.miniwob.labeled_demonstration import \
        LabeledDemonstration, WeightedProgram
from wge.miniwob.fields import Fields


# Register new Oracles by defining them below and adding them to the
# get_oracle function
def get_program_oracle(subdomain, config):
    # TODO (evan): None of these oracles work right now.
    #if subdomain == "click-tab-2":
    #    return ClickTab2OracleProgramPolicy(config)
    #elif subdomain == "click-tab-2-easy":
    #    return ClickTabEasyOracleProgramPolicy(config)
    #elif subdomain == "click-tab-2-medium":
    #    return ClickTabMediumOracleProgramPolicy(config)
    #elif subdomain == "click-button":
    #    return ClickButtonOracleProgramPolicy(config)
    #elif subdomain == "login-user":
    #    return LoginUserOracleProgramPolicy(config)
    #elif subdomain == "email-inbox-reply":
    #    return EmailInboxReplyOracleProgramPolicy(config)
    #elif subdomain == "email-inbox-forward":
    #    return EmailInboxForwardOracleProgramPolicy(config)
    if subdomain == "click-checkboxes":
        return ClickCheckboxesOracle(config)
    elif subdomain == "social-media":
        return SocialMediaOracle(config)
    else:
        raise ValueError("No program oracle policy for {}".format(subdomain))


class ClickTab2OracleProgramPolicy(LinearProgramPolicy):
    def __init__(self, config):
        # Appears on first page
        labeled_demos = [LabeledDemonstration.from_oracle_programs(
            [[WeightedProgram(
                ClickToken(LikeToken(UtteranceSelectorToken(12, 13))), 1)]],
            "Switch between the tabs to find and click on the link \"x\".",
            Fields({"target": "x"}))]

        # Appears in other tabs
        labeled_demos += [LabeledDemonstration.from_oracle_programs(
            [[WeightedProgram(
                ClickToken(LikeToken(StringToken(u"Tab #2"))), 1)],
             [WeightedProgram(
                 ClickToken(LikeToken(UtteranceSelectorToken(12, 13))), 1)]],
            "Switch between the tabs to find and click on the link \"x\".",
            Fields({"target": "x"}))]
        labeled_demos += [LabeledDemonstration.from_oracle_programs(
            [[WeightedProgram(
                ClickToken(LikeToken(StringToken(u"Tab #2"))), 1)],
             [WeightedProgram(
                ClickToken(LikeToken(StringToken(u"Tab #3"))), 1)],
             [WeightedProgram(
                 ClickToken(LikeToken(UtteranceSelectorToken(12, 13))), 1)]],
            "Switch between the tabs to find and click on the link \"x\".",
            Fields({"target": "x"}))]

        super(ClickTab2OracleProgramPolicy, self).__init__(labeled_demos, config)


class ClickTabEasyOracleProgramPolicy(LinearProgramPolicy):
    def __init__(self, config):
        # Appears on first page
        labeled_demos = [LabeledDemonstration.from_oracle_programs(
            [[WeightedProgram(
                ClickToken(LikeToken(UtteranceSelectorToken(12, 13))), 1)]],
            "Switch between the tabs to find and click on the link \"x\".",
            Fields({"target": "x"}))]

        super(ClickTabEasyOracleProgramPolicy, self).__init__(labeled_demos, config)


class ClickTabMediumOracleProgramPolicy(LinearProgramPolicy):
    def __init__(self, config):
        # Appears on first page
        labeled_demos = [LabeledDemonstration.from_oracle_programs(
            [[WeightedProgram(
                ClickToken(LikeToken(UtteranceSelectorToken(12, 13))), 1)]],
            "Switch between the tabs to find and click on the link \"x\".",
            Fields({"target": "x"}))]
        labeled_demos += [LabeledDemonstration.from_oracle_programs(
            [[WeightedProgram(
                ClickToken(LikeToken(StringToken(u"Tab #2"))), 1)]],
            "Switch between the tabs to find and click on the link \"x\".",
            Fields({"target": "x"}))]

        super(ClickTabMediumOracleProgramPolicy, self).__init__(labeled_demos, config)


class ClickButtonOracleProgramPolicy(LinearProgramPolicy):
    def __init__(self, config):
        labeled_demos = [LabeledDemonstration.from_oracle_programs(
            [[WeightedProgram(
                ClickToken(LikeToken(UtteranceSelectorToken(4, 5))), 1)]],
            "Switch between the tabs to find and click on the link \"x\".",
            Fields({"target": "x"}))]

        super(ClickButtonOracleProgramPolicy, self).__init__(labeled_demos, config)


class LoginUserOracleProgramPolicy(LinearProgramPolicy):
    def __init__(self, config):
        labeled_demos = [LabeledDemonstration.from_oracle_programs(
            [[WeightedProgram(
                FocusAndTypeToken(NearToken(LikeToken(
                    StringToken(u"Username"))), UtteranceSelectorToken(4, 5)),
                1)],
             [WeightedProgram(
                FocusAndTypeToken(NearToken(LikeToken(StringToken(
                    u"Password"))), UtteranceSelectorToken(10, 11)), 1)],
             [WeightedProgram(
                ClickToken(LikeToken(StringToken(u"Login"))), 1)]],
            "Enter the username \"blah\" and the password \"blah\" into the text fields and press login.",
            Fields({"username": "blah", "password": "blah"}))]

        super(LoginUserOracleProgramPolicy, self).__init__(labeled_demos, config)


class EmailInboxReplyOracleProgramPolicy(LinearProgramPolicy):
    def __init__(self, config):
        labeled_demos = [LabeledDemonstration.from_oracle_programs(
            [[WeightedProgram(
                ClickToken(LikeToken(FieldsValueSelectorToken(1))), 1)],
             [WeightedProgram(
                ClickToken(LikeToken(StringToken(u"Reply"))), 1)],
             [WeightedProgram(
                FocusAndTypeToken(
                    InputElementsToken(), FieldsValueSelectorToken(0)), 1)],
             [WeightedProgram(
                 ClickToken(LikeToken(StringToken(u""))), 1)]], # TODO: Remove this hack
            "Find the email by Harmonia and reply to them with the text \"hello\".",
            Fields({"by": "Harmonia", "message": "hello"}))]

        super(EmailInboxReplyOracleProgramPolicy, self).__init__(labeled_demos, config)


class EmailInboxForwardOracleProgramPolicy(LinearProgramPolicy):
    def __init__(self, config):
        labeled_demos = [LabeledDemonstration.from_oracle_programs(
            [[WeightedProgram(
                ClickToken(LikeToken(FieldsValueSelectorToken(1))), 1)],
             [WeightedProgram(
                ClickToken(LikeToken(StringToken(u"Forward"))), 1)],
             [WeightedProgram(
                FocusAndTypeToken(
                    InputElementsToken(), FieldsValueSelectorToken(0)), 1)],
             [WeightedProgram(
                 ClickToken(LikeToken(StringToken(u""))), 1)]], # TODO: Remove this hack
            "Find the email by Harmonia and forward that email to Eleanore.",
            Fields({"by": "Harmonia", "to": "Eleanore"}))]

        super(EmailInboxForwardOracleProgramPolicy, self).__init__(labeled_demos, config)


class EmailNoScrollOracle(LinearProgramPolicy):
    def __init__(self, config):
        labeled_demos = [
            LabeledDemonstration.from_oracle_programs(
                [
                    #######
                    # forward

                    # Click name of sender
                    [WeightedProgram(ClickToken(
                        LikeToken(FieldsValueSelectorToken(0))), 1)],
                    # Forward
                    [WeightedProgram(
                       ClickToken(LikeToken(StringToken(u"Forward"))), 1)],
                    # Type name of recipient
                    [WeightedProgram(FocusAndTypeToken(
                        InputElementsToken(),
                        FieldsValueSelectorToken(2)), 1)],
                    # Send button
                    [WeightedProgram(
                        ClickToken(LikeToken(StringToken(u""))), 1)]
                ],
                "Find the email by A and forward that email to B.",
                Fields({"by": "A", "task": "forward", "to": "B"})),

            LabeledDemonstration.from_oracle_programs(
                [
                    #######
                    # reply

                    # Click name of sender
                    [WeightedProgram(ClickToken(
                        LikeToken(FieldsValueSelectorToken(0))), 1)],
                    # Reply
                    [WeightedProgram(
                       ClickToken(LikeToken(StringToken(u"Reply"))), 1)],
                    # Type message
                    [WeightedProgram(FocusAndTypeToken(
                        InputElementsToken(),
                        FieldsValueSelectorToken(1)), 1)],
                    # Send button
                    [WeightedProgram(
                        ClickToken(LikeToken(StringToken(u""))), 1)]
                ],
                "Find the email by A and reply to them with the text \"B\".",
                Fields({"by": "A", "message": "B", "task": "reply"})),

            LabeledDemonstration.from_oracle_programs(
                [
                    #######
                    # trash

                    # Click name of sender
                    [WeightedProgram(ClickToken(SameRowToken(
                        LikeToken(FieldsValueSelectorToken(0)))), 1)],
                ],
                "Find the email by A and click the trash icon to delete it.",
                Fields({"by": "A", "task": "delete"})),

            LabeledDemonstration.from_oracle_programs(
                [
                    #######
                    # star

                    # Click name of sender
                    [WeightedProgram(ClickToken(SameRowToken(
                            LikeToken(FieldsValueSelectorToken(0)))), 1)],
                ],
                ("Find the email by A and click the star "
                 "icon to mark it as important."),
                Fields({"by": "A", "task": "star"})),
        ]
        super(EmailNoScrollOracle, self).__init__(labeled_demos, config)


class SocialMediaOracle(LinearProgramPolicy):
    def __init__(self, config):
        labeled_demos = [
            # block
            LabeledDemonstration.from_oracle_programs(
                [
                    # More
                    [WeightedProgram(ClickToken(NearToken(
                        LikeToken(FieldsValueSelectorToken(1)),
                        classes="more")), 1)],
                    # Block
                    [WeightedProgram(
                       ClickToken(LikeToken(StringToken(u"Block"))), 1)]
                ],
                "For the user @jess, click on the \"Block\" button.",
                Fields({"user": "@jess", "button": "block"})),

            # reply
            LabeledDemonstration.from_oracle_programs(
                [
                    # reply
                    [WeightedProgram(ClickToken(NearToken(
                        LikeToken(FieldsValueSelectorToken(1)),
                        classes="reply")), 1)],
                ],
                "For the user @jess, click on the \"Reply\" button.",
                Fields({"user": "@jess", "button": "reply"})),

            # like
            LabeledDemonstration.from_oracle_programs(
                [
                    # like
                    [WeightedProgram(ClickToken(NearToken(
                        LikeToken(FieldsValueSelectorToken(1)),
                        classes="like")), 1)],
                ],
                "For the user @jess, click on the \"Like\" button.",
                Fields({"user": "@jess", "button": "like"})),

            # share via DM
            LabeledDemonstration.from_oracle_programs(
                [
                    # More
                    [WeightedProgram(ClickToken(NearToken(
                        LikeToken(FieldsValueSelectorToken(1)),
                        classes="more")), 1)],
                    # share
                    [WeightedProgram(
                       ClickToken(LikeToken(StringToken(u"share"))), 1)]
                ],
                "For the user @jess, click on the \"share\" button.",
                Fields({"user": "@jess", "button": "share"})),

            # copy
            LabeledDemonstration.from_oracle_programs(
                [
                    # More
                    [WeightedProgram(ClickToken(NearToken(
                        LikeToken(FieldsValueSelectorToken(1)),
                        classes="more")), 1)],
                    # copy
                    [WeightedProgram(
                       ClickToken(LikeToken(StringToken(u"copy"))), 1)]
                ],
                "For the user @jess, click on the \"Copy\" button.",
                Fields({"user": "@jess", "button": "copy"})),

            # embed
            LabeledDemonstration.from_oracle_programs(
                [
                    # More
                    [WeightedProgram(ClickToken(NearToken(
                        LikeToken(FieldsValueSelectorToken(1)),
                        classes="more")), 1)],
                    # embed
                    [WeightedProgram(
                       ClickToken(LikeToken(StringToken(u"embed"))), 1)]
                ],
                "For the user @jess, click on the \"Embed\" button.",
                Fields({"user": "@jess", "button": "embed"})),

            # mute
            LabeledDemonstration.from_oracle_programs(
                [
                    # More
                    [WeightedProgram(ClickToken(NearToken(
                        LikeToken(FieldsValueSelectorToken(1)),
                        classes="more")), 1)],
                    # mute
                    [WeightedProgram(
                       ClickToken(LikeToken(StringToken(u"mute"))), 1)]
                ],
                "For the user @jess, click on the \"Mute\" button.",
                Fields({"user": "@jess", "button": "mute"})),

            # report
            LabeledDemonstration.from_oracle_programs(
                [
                    # More
                    [WeightedProgram(ClickToken(NearToken(
                        LikeToken(FieldsValueSelectorToken(1)),
                        classes="more")), 1)],
                    # embed
                    [WeightedProgram(
                       ClickToken(LikeToken(StringToken(u"report"))), 1)]
                ],
                "For the user @jess, click on the \"Report\" button.",
                Fields({"user": "@jess", "button": "report"})),

            # retweet
            LabeledDemonstration.from_oracle_programs(
                [
                    # like
                    [WeightedProgram(ClickToken(NearToken(
                        LikeToken(FieldsValueSelectorToken(1)),
                        classes="retweet")), 1)],
                ],
                "For the user @jess, click on the \"Retweet\" button.",
                Fields({"user": "@jess", "button": "retweet"})),
        ]

        super(SocialMediaOracle, self).__init__(labeled_demos, config)


class ClickCheckboxesOracle(LinearProgramPolicy):
    def __init__(self, config):
        submit = lambda: [WeightedProgram(ClickToken(LikeToken(StringToken(u"Submit"))), 1)]
        check_field = lambda i: [WeightedProgram(ClickToken(SameRowToken(LikeToken(FieldsValueSelectorToken(i)))), 1)]

        def demo(num_boxes, randomize_order):
            programs_sequence = [check_field(i + 1) for i in range(num_boxes)]

            if randomize_order:
                random.shuffle(programs_sequence)

            programs_sequence.append(submit())

            utt, fields = {
                0: ("Click nothing then submit",
                    Fields({"button": "submit"})),
                1: ("Click a then submit",
                    Fields({"target 0": "a", "button": "submit"})),
                2: ("Click a,b then submit",
                    Fields({"target 0": "a", "target 1": "b", "button": "submit"})),
                3: ("Click a,b,c then submit",
                    Fields({
                        "target 0": "a", "target 1": "b", "target 2": "c",
                        "button": "submit"})),
                4: ("Click a,b,c,d then submit",
                    Fields({
                        "target 0": "a", "target 1": "b", "target 2": "c",
                        "target 3": "d", "button": "submit"})),
                5: ("Click a,b,c,d,e then submit",
                    Fields({
                        "target 0": "a", "target 1": "b", "target 2": "c",
                        "target 3": "d", "target 4": "e", "button": "submit"})),
                6: ("Click a,b,c,d,e,f then submit",
                    Fields({
                        "target 0": "a", "target 1": "b", "target 2": "c",
                        "target 3": "d", "target 4": "e", "target 5": "f",
                        "button": "submit"})),
            }[num_boxes]

            return LabeledDemonstration.from_oracle_programs(programs_sequence, utt, fields)

        # randomize = True
        # num_samples = 100
        randomize = False
        num_samples = 1
        
        labeled_demos = []
        for i in range(7):
            demos_for_i = [demo(i, randomize) for _ in range(num_samples)]
            labeled_demos.extend(demos_for_i)

        super(ClickCheckboxesOracle, self).__init__(labeled_demos, config)

    def update_from_episodes(self, episodes, gamma, take_grad_step):
        return  # don't do any updates
