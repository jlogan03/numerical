/// High-level example area selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ExampleId {
    /// Landing page and catalog overview.
    Home,
    /// Continuous-time process-model exploration.
    LtiProcessModels,
    /// Digital IIR filter design exploration.
    FilterDesign,
    /// Estimation placeholder.
    Estimation,
    /// Identification placeholder.
    Identification,
    /// Model-reduction placeholder.
    Reduction,
    /// Controller-synthesis placeholder.
    Synthesis,
}

/// Whether an example page is already implemented or is only scaffolded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExampleStatus {
    /// Backed by a working interactive example page.
    Ready,
    /// Present in the app shell but still a placeholder.
    Scaffolded,
}

impl ExampleStatus {
    /// Human-readable status label for badges in the UI shell.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Ready => "ready",
            Self::Scaffolded => "scaffolded",
        }
    }

    /// CSS class for the corresponding badge.
    #[must_use]
    pub const fn class_name(self) -> &'static str {
        match self {
            Self::Ready => "ready",
            Self::Scaffolded => "scaffolded",
        }
    }
}

/// Metadata for one selectable example page.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExampleEntry {
    /// Stable identifier used by the app shell.
    pub id: ExampleId,
    /// Display title.
    pub title: &'static str,
    /// Short summary shown in navigation and cards.
    pub summary: &'static str,
    /// Implementation status badge.
    pub status: ExampleStatus,
}

/// Metadata for a major feature area in the interactive example browser.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExampleGroup {
    /// Display title for the feature area.
    pub title: &'static str,
    /// Group-level summary for the home page.
    pub summary: &'static str,
    /// Example pages that fall under this feature area.
    pub entries: &'static [ExampleEntry],
}

/// Example groups exposed by the current app shell.
pub const EXAMPLE_GROUPS: &[ExampleGroup] = &[
    ExampleGroup {
        title: "LTI Analysis",
        summary: "Interactive sampling and plotting for process models, time responses, and frequency views.",
        entries: &[ExampleEntry {
            id: ExampleId::LtiProcessModels,
            title: "Process Models",
            summary: "Explore delayed first-order process models with step and Bode views driven by `numerical`.",
            status: ExampleStatus::Ready,
        }],
    },
    ExampleGroup {
        title: "Filter Design",
        summary: "Digital filter design examples with immediately inspectable frequency-domain behavior.",
        entries: &[ExampleEntry {
            id: ExampleId::FilterDesign,
            title: "Butterworth Lowpass",
            summary: "Tune order, cutoff, and sample rate and inspect the designed IIR directly in SOS form.",
            status: ExampleStatus::Ready,
        }],
    },
    ExampleGroup {
        title: "Estimation",
        summary: "Kalman, EKF, and UKF examples that can show covariance evolution, residuals, and convergence.",
        entries: &[ExampleEntry {
            id: ExampleId::Estimation,
            title: "Estimator Workbench",
            summary: "Placeholder for linear and nonlinear estimator examples with split-step and monolithic flows.",
            status: ExampleStatus::Scaffolded,
        }],
    },
    ExampleGroup {
        title: "Identification",
        summary: "Data-driven realization and Markov-parameter recovery examples for ERA and OKID workflows.",
        entries: &[ExampleEntry {
            id: ExampleId::Identification,
            title: "ERA / OKID",
            summary: "Placeholder for identification examples that start from sampled IO data and recover models.",
            status: ExampleStatus::Scaffolded,
        }],
    },
    ExampleGroup {
        title: "Reduction",
        summary: "Interactive views of HSVs, retained order, and reduced-model fidelity under truncation choices.",
        entries: &[ExampleEntry {
            id: ExampleId::Reduction,
            title: "Balanced Truncation",
            summary: "Placeholder for reduction examples comparing full and reduced-order responses.",
            status: ExampleStatus::Scaffolded,
        }],
    },
    ExampleGroup {
        title: "Synthesis",
        summary: "Controller-design examples spanning PID, LQR, LQG, and pole placement.",
        entries: &[ExampleEntry {
            id: ExampleId::Synthesis,
            title: "Controller Design",
            summary: "Placeholder for closed-loop design examples with gain tuning and response comparison.",
            status: ExampleStatus::Scaffolded,
        }],
    },
];

impl ExampleId {
    /// Returns the catalog metadata for this example identifier.
    #[must_use]
    pub const fn entry(self) -> ExampleEntry {
        match self {
            Self::Home => ExampleEntry {
                id: Self::Home,
                title: "Overview",
                summary: "Start here to browse the major feature areas and implemented demos.",
                status: ExampleStatus::Ready,
            },
            Self::LtiProcessModels => EXAMPLE_GROUPS[0].entries[0],
            Self::FilterDesign => EXAMPLE_GROUPS[1].entries[0],
            Self::Estimation => EXAMPLE_GROUPS[2].entries[0],
            Self::Identification => EXAMPLE_GROUPS[3].entries[0],
            Self::Reduction => EXAMPLE_GROUPS[4].entries[0],
            Self::Synthesis => EXAMPLE_GROUPS[5].entries[0],
        }
    }
}
