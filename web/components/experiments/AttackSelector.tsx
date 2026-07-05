"use client";

import { Badge } from "@/components/ui/badge";
import { Checkbox } from "@/components/ui/checkbox";
import type { AttackInfo } from "@/lib/types";

export function AttackSelector({
  attacks,
  selected,
  onChange,
}: {
  attacks: AttackInfo[];
  selected: string[];
  onChange: (attacks: string[]) => void;
}) {
  const set = new Set(selected);

  function toggle(name: string) {
    const next = new Set(set);
    if (next.has(name)) next.delete(name);
    else next.add(name);
    onChange([...next]);
  }

  return (
    <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
      {attacks.map((attack) => (
        <label key={attack.name} className="flex items-start gap-2 text-sm" title={attack.description}>
          <Checkbox checked={set.has(attack.name)} onCheckedChange={() => toggle(attack.name)} />
          <span>
            {attack.name.replaceAll("_", " ")}
            {attack.changes_length_or_rate && (
              <Badge
                variant="secondary"
                className="ml-1.5"
                title="Changes signal length or sample rate; decodes usually fail (that is the point of the test)"
              >
                destructive
              </Badge>
            )}
          </span>
        </label>
      ))}
    </div>
  );
}
