"use client";

import { useRef, useState } from "react";
import { Loader2, Upload } from "lucide-react";

import { PageHeader } from "@/components/layout/PageHeader";
import { useCatalog } from "@/components/experiments/useCatalog";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { api } from "@/lib/api";

export default function DatasetsSettingsPage() {
  const { catalog, error, refresh } = useCatalog();
  const [name, setName] = useState("");
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  async function upload(files: FileList | null) {
    if (!files || files.length === 0) return;
    setUploading(true);
    setUploadError(null);
    try {
      await api.uploadDataset(name, [...files]);
      setName("");
      refresh();
    } catch (err) {
      setUploadError((err as Error).message);
    } finally {
      setUploading(false);
      if (inputRef.current) inputRef.current.value = "";
    }
  }

  return (
    <div className="space-y-6">
      <PageHeader
        title="Datasets"
        description="Packaged corpora shipped with the taf package plus datasets you upload. Research Experiment additionally accepts a local directory path on the backend machine."
      />
      {error && <p className="text-sm text-destructive">{error}</p>}

      <Card>
        <CardHeader>
          <CardTitle>Upload a dataset</CardTitle>
          <CardDescription>
            WAV, FLAC or OGG files. Uploads live in the API's temp storage for this session and
            appear as <code>upload:&lt;id&gt;</code> datasets in every experiment.
          </CardDescription>
        </CardHeader>
        <CardContent className="flex flex-wrap items-center gap-2">
          <Input
            placeholder="dataset name (optional)"
            className="w-56"
            value={name}
            onChange={(event) => setName(event.target.value)}
          />
          <input
            ref={inputRef}
            type="file"
            accept=".wav,.flac,.ogg"
            multiple
            className="hidden"
            onChange={(event) => upload(event.target.files)}
          />
          <Button
            type="button"
            variant="outline"
            disabled={uploading}
            onClick={() => inputRef.current?.click()}
          >
            {uploading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Upload className="h-4 w-4" />}
            Choose files
          </Button>
          {uploadError && <p className="w-full text-xs text-destructive">{uploadError}</p>}
        </CardContent>
      </Card>

      <Card>
        <CardContent className="pt-6">
          {!catalog ? (
            <Skeleton className="h-40 w-full" />
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Dataset</TableHead>
                  <TableHead>Kind</TableHead>
                  <TableHead>Files</TableHead>
                  <TableHead>Formats</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {catalog.datasets.map((dataset) => (
                  <TableRow key={dataset.id}>
                    <TableCell className="whitespace-nowrap font-medium">
                      {dataset.kind === "uploaded" ? `${dataset.label} (${dataset.id})` : dataset.id}
                    </TableCell>
                    <TableCell>
                      <Badge variant={dataset.kind === "uploaded" ? "default" : "secondary"}>
                        {dataset.kind}
                      </Badge>
                    </TableCell>
                    <TableCell>{dataset.file_count}</TableCell>
                    <TableCell className="uppercase text-xs text-muted-foreground">
                      {dataset.formats.join(", ")}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
