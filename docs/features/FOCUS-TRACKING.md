# Focus & Attention Tracking

Comprehensive focus tracking system with AI-powered insights, distraction monitoring, and productivity analytics.

## Overview

The Focus Tracking system helps you understand your attention patterns, identify distractions, and optimize your work sessions for maximum productivity. It provides:

- **Quality Scoring**: AI-powered analysis of session effectiveness
- **Flow State Detection**: Identifies when you achieve optimal focus
- **Distraction Monitoring**: Tracks interruptions and patterns
- **Peak Hours Analysis**: Finds your most productive times
- **Personalized Recommendations**: AI-driven improvement suggestions

## Quick Start

### Starting a Focus Session

```bash
# Basic session
python -m src.vega.core.cli productivity focus-start

# With specific type
python -m src.vega.core.cli productivity focus-start --type deep_work

# Link to a task
python -m src.vega.core.cli productivity focus-start --task-id task-123

# Add context
python -m src.vega.core.cli productivity focus-start --context "Feature development"
```

### Recording Interruptions

```bash
# Record notification
python -m src.vega.core.cli productivity focus-interruption notification "Slack message"

# With duration and impact
python -m src.vega.core.cli productivity focus-interruption distraction "Social media" \
  --duration 120 --impact 0.7
```

### Ending a Session

```bash
# Basic completion
python -m src.vega.core.cli productivity focus-stop

# With notes
python -m src.vega.core.cli productivity focus-stop --notes "Completed feature implementation"
```

## Session Types

Choose the appropriate type for each focus session:

| Type | Description | Optimal Duration | Best For |
|------|-------------|------------------|----------|
| **deep_work** | Uninterrupted focused work | 60-90 min | Complex problem-solving, coding |
| **shallow_work** | Less demanding tasks | 25-45 min | Email, admin tasks, reviews |
| **learning** | Study and skill development | 45-60 min | Tutorials, documentation, courses |
| **creative** | Creative/brainstorming work | 30-60 min | Design, writing, ideation |
| **meeting** | Collaborative sessions | 30-60 min | Meetings, pair programming |
| **break** | Rest and recovery | 5-15 min | Breaks between sessions |

## Interruption Types

Track different kinds of distractions:

- **notification**: Emails, messages, alerts
- **distraction**: Self-initiated breaks (social media, browsing)
- **external**: Unexpected events (calls, visitors)
- **context_switch**: Shifting between tasks
- **break**: Planned breaks

## Quality Scoring

Sessions are scored 0.0-1.0 based on:

### Duration Factor (50%)

- **Optimal Range**: 25-90 minutes
- Too short (<10 min): 0.3 score
- Short (10-25 min): 0.6 score
- Optimal (25-90 min): 1.0 score
- Long (90-120 min): 0.9 score
- Very long (>120 min): 0.7 score (diminishing returns)

### Interruption Factor (50%)

- **None**: 1.0 score (perfect focus)
- **Low Impact** (1-2 minor): 0.9 score
- **Moderate** (3-5 or high-impact): 0.6 score
- **High** (>5 interruptions): 0.3 score

### Time-of-Day Bonus

- **Peak Hours** (9-11 AM, 2-4 PM): +0.1 bonus
- **Good Hours** (8 AM, 11 AM, 1 PM, 4 PM): +0.05 bonus
- **Other Hours**: No bonus

### Type Multiplier

- **deep_work**: 1.2x multiplier
- **creative**: 1.15x multiplier
- **learning**: 1.1x multiplier
- **shallow_work**: 0.9x multiplier
- **meeting**: 0.7x multiplier
- **break**: 0.5x multiplier

### Quality Ratings

- **0.85-1.00**: 🌟 Excellent focus!
- **0.70-0.84**: ✅ Good session
- **0.50-0.69**: ⚠️ Could be better
- **<0.50**: ❌ Needs improvement

## Flow State Detection

Flow state is achieved when:

- Quality score ≥ 0.85
- ≤2 interruptions
- Duration ≥20 minutes

## Viewing Analytics

### 7-Day Metrics

```bash
python -m src.vega.core.cli productivity focus-metrics --days 7
```

Shows:

- Total sessions and focus time
- Average duration and quality
- Deep work percentage
- Interruption count
- Quality trend (↗️ improving, ➡️ stable, ↘️ declining)
- Peak focus hours
- Best performing day

### Session History

```bash
# Recent sessions
python -m src.vega.core.cli productivity focus-history --limit 10

# Filter by task
python -m src.vega.core.cli productivity focus-history --task-id task-123

# Filter by type
python -m src.vega.core.cli productivity focus-history --type deep_work
```

### AI-Powered Insights

```bash
python -m src.vega.core.cli productivity focus-insights
```

Provides:

- **Peak Focus Hours**: Your most productive time blocks
- **Optimal Session Length**: Personalized duration recommendations
- **Distraction Patterns**: Common interruption sources and times
- **Improvement Recommendations**: Actionable suggestions
- **Quality Trends**: Week-over-week comparisons

### Reports

```bash
# Weekly summary
python -m src.vega.core.cli productivity focus-report --type weekly

# All-time summary
python -m src.vega.core.cli productivity focus-report --type summary

# Last 30 days
python -m src.vega.core.cli productivity focus-report --type summary --days 30
```

### All-Time Statistics

```bash
python -m src.vega.core.cli productivity focus-stats
```

Shows:

- Total sessions by type
- Average quality by type
- Total focus time
- Flow state sessions
- Quality distribution histogram

## Example Workflows

### Deep Work Session

```bash
# Morning deep work block
python -m src.vega.core.cli productivity focus-start \
  --type deep_work \
  --task-id feature-123 \
  --context "Implementing authentication system"

# Work for 90 minutes...
# If interrupted:
python -m src.vega.core.cli productivity focus-interruption notification "Email alert" \
  --duration 30 --impact 0.3

# Complete session
python -m src.vega.core.cli productivity focus-stop \
  --notes "Completed OAuth integration"

# Review metrics
python -m src.vega.core.cli productivity focus-metrics
```

### Learning Session

```bash
# Start learning session
python -m src.vega.core.cli productivity focus-start \
  --type learning \
  --context "React hooks tutorial"

# Study for 45 minutes...

# End session
python -m src.vega.core.cli productivity focus-stop \
  --notes "Completed useState and useEffect modules"
```

### Task-Linked Workflow

```bash
# Create task
python -m src.vega.core.cli productivity task-create \
  "Implement user dashboard" \
  --project vega-ui

# Start focused work on task
python -m src.vega.core.cli productivity focus-start \
  --task-id <task-id-from-above> \
  --type deep_work

# Work session...

# Complete and review
python -m src.vega.core.cli productivity focus-stop
python -m src.vega.core.cli productivity focus-history --task-id <task-id>
```

## Distraction Management

### Pattern Analysis

The system automatically analyzes your interruption patterns:

```bash
# View insights including distraction analysis
python -m src.vega.core.cli productivity focus-insights
```

Common patterns detected:

- **Notification Overload**: >5 notifications per session
- **Frequent Context Switching**: Multiple task switches
- **Habitual Distractions**: Recurring self-initiated breaks
- **Peak Distraction Times**: Specific hours with high interruptions

### Mitigation Strategies

Based on detected patterns, you'll receive suggestions like:

**For Notification Overload:**

- Enable Do Not Disturb during focus sessions
- Batch check notifications at scheduled times
- Disable non-critical app notifications

**For Context Switching:**

- Use single-tasking approach
- Plan context switch buffers
- Complete tasks before switching

**For Time-Based Patterns:**

- Schedule focus work during low-distraction hours
- Communicate "focus hours" to team
- Use morning hours for deep work

## Integration with Task Manager

Focus tracking seamlessly integrates with the Task Manager:

### Linking Sessions to Tasks

```bash
# Start session linked to task
python -m src.vega.core.cli productivity focus-start --task-id task-123
```

### View Task Focus Time

```bash
# See all focus sessions for a task
python -m src.vega.core.cli productivity focus-history --task-id task-123

# Calculate total focus time
python -m src.vega.core.cli productivity task-view task-123
```

### Task Quality Insights

Track which tasks generate the highest quality focus:

- High-quality task patterns
- Task types requiring most focus
- Optimal times for specific task types

## Best Practices

### Session Planning

1. **Choose Appropriate Types**: Match session type to work nature
2. **Set Realistic Durations**: Aim for 25-90 minute blocks
3. **Schedule Peak Hours**: Use insights to plan important work
4. **Link to Tasks**: Track time investment per task

### During Sessions

1. **Record All Interruptions**: Build accurate patterns
2. **Note Impact Levels**: Help AI learn what affects you most
3. **Identify Sources**: Track where distractions come from
4. **Stay Honest**: Accurate logging = better insights

### After Sessions

1. **Add Notes**: Document what worked/didn't work
2. **Review Quality**: Understand score factors
3. **Check Insights**: Act on recommendations
4. **Track Trends**: Monitor improvement over time

### Weekly Review

```bash
# Generate weekly report
python -m src.vega.core.cli productivity focus-report --type weekly

# Review insights
python -m src.vega.core.cli productivity focus-insights

# Check all-time stats
python -m src.vega.core.cli productivity focus-stats
```

## Tips for Better Focus

### Environment Setup

- Minimize visible distractions
- Use physical "focus mode" indicators
- Keep workspace organized
- Control ambient noise levels

### Digital Hygiene

- Close unnecessary apps/tabs
- Enable Do Not Disturb
- Use website blockers during deep work
- Schedule communication batches

### Physical Well-being

- Take regular breaks (tracked as sessions!)
- Stay hydrated
- Maintain good posture
- Get adequate sleep

### Mental Preparation

- Clear task list before starting
- Set specific session goals
- Use warm-up rituals
- Practice single-tasking

## Data Storage

Focus tracking data is stored in `~/.vega/focus/`:

- **sessions.json**: All focus sessions
- **interruptions.json**: Interruption records
- **metrics_cache.json**: Cached analytics

### Backup Recommendations

```bash
# Backup focus data
cp -r ~/.vega/focus ~/.vega/focus-backup-$(date +%Y%m%d)

# Restore from backup
cp -r ~/.vega/focus-backup-20250120 ~/.vega/focus
```

## Advanced Features

### Custom Analysis

Use the Python API for custom analytics:

```python
from src.vega.productivity.focus_tracker import FocusTracker
from datetime import date, timedelta

tracker = FocusTracker()

# Get sessions for analysis
sessions = tracker.get_session_history(limit=100)

# Custom calculations
total_deep_work = sum(
    s.duration for s in sessions 
    if s.session_type == FocusType.DEEP_WORK
)

# Advanced insights
from src.vega.productivity.focus_tracker import ProductivityInsights
insights = ProductivityInsights()
peak_hours = insights.get_peak_focus_hours(sessions)
recommendations = insights.get_improvement_recommendations(sessions, metrics)
```

### Programmatic Access

Integrate focus tracking into your workflows:

```python
from src.vega.productivity.focus_tracker import (
    FocusTracker,
    FocusType,
    InterruptionType,
)

tracker = FocusTracker()

# Start automated session tracking
session_id = tracker.start_session(
    task_id="automated-task",
    focus_type=FocusType.DEEP_WORK,
)

# Your work...

# End session
session = tracker.end_session(session_id)
print(f"Quality: {session.quality_score}")
```

## Troubleshooting

### Session Not Ending

```bash
# Check active session
python -m src.vega.core.cli productivity focus-metrics

# Force end via Python
python -c "from src.vega.productivity.focus_tracker import FocusTracker; \
  t = FocusTracker(); \
  s = t.get_active_session(); \
  t.end_session(s.session_id) if s else print('No active session')"
```

### Missing Data

Check storage files exist:

```bash
ls -lh ~/.vega/focus/
```

Verify JSON validity:

```bash
python -m json.tool ~/.vega/focus/sessions.json > /dev/null && echo "Valid" || echo "Invalid"
```

### Low Quality Scores

Common causes:

1. **Too Many Interruptions**: Enable focus mode
2. **Wrong Session Type**: Choose appropriate type
3. **Suboptimal Duration**: Aim for 25-90 minutes
4. **Poor Timing**: Work during peak hours

## FAQ

**Q: What's a good quality score?**  
A: 0.70+ is good, 0.85+ is excellent. Focus on consistency over perfection.

**Q: How long should focus sessions be?**  
A: 25-90 minutes is optimal. Start with 25-minute blocks if new to focus work.

**Q: Should I track breaks?**  
A: Yes! Breaks are important. Use `--type break` to track recovery time.

**Q: How do interruptions affect quality?**  
A: Impact depends on duration and type. Quick checks (<30s) have minimal impact; long context switches (>2min) significantly reduce quality.

**Q: Can I edit past sessions?**  
A: Currently no direct editing. Use notes field to add context or corrections.

**Q: How often should I review insights?**  
A: Weekly reviews are recommended. Check metrics daily, insights weekly, full report monthly.

**Q: Does focus tracking work offline?**  
A: Yes! All data is stored locally. No internet required.

**Q: Can I export my data?**  
A: Yes, data is stored as JSON in `~/.vega/focus/`. Easy to backup or analyze externally.

## Related Documentation

- [Task Manager](TASK-MANAGEMENT.md) - Link focus sessions to tasks
- [Knowledge Base](KNOWLEDGE-BASE.md) - Document focus insights
- [Architecture](ARCHITECTURE.md) - Technical implementation details
- [Roadmap](../../roadmap.md) - Future focus tracking features

## Support

For issues or feature requests:

1. Check logs: `~/.vega/logs/productivity.log`
2. Review test suite: `tests/productivity/test_focus_tracker.py`
3. See integration tests: `tests/productivity/test_focus_integration.py`
4. Open GitHub issue with details

---

### Happy Focusing! 🎯
