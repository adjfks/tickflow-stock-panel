import { useEffect, useMemo, useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { BarChart3, ChevronLeft, ChevronRight, LoaderCircle, Search, X } from 'lucide-react'
import { Modal } from '@/components/Modal'
import { DatePicker } from '@/components/DatePicker'
import {
  api,
  type ScreenerStrategy,
  type StrategyWinRateDetail,
  type StrategyWinRateResult,
  type WinRateCombinationKey,
} from '@/lib/api'

interface Props {
  open: boolean
  strategies: ScreenerStrategy[]
  minDate: string
  maxDate: string
  defaultStart: string
  defaultEnd: string
  onClose: () => void
}

const BOARD_OPTIONS = ['沪主板', '深主板', '创业板', '科创板', '北交所']
const COMBINATION_KEYS: WinRateCombinationKey[] = ['open_open', 'open_close', 'close_open', 'close_close']
const PAGE_SIZE = 50

function percent(value: number | null | undefined): string {
  return value == null ? '-' : `${(value * 100).toFixed(2)}%`
}

function skipReasons(stats: StrategyWinRateResult['summary']['combinations'][WinRateCombinationKey]): string {
  return Object.entries(stats.skip_reasons ?? {})
    .map(([reason, count]) => `${reason} ${count}`)
    .join(' · ')
}

function dateBefore(value: string, days: number): string {
  if (!value) return ''
  const date = new Date(`${value}T00:00:00Z`)
  date.setUTCDate(date.getUTCDate() - days)
  return date.toISOString().slice(0, 10)
}

function ReturnCell({ detail, combination }: { detail: StrategyWinRateDetail; combination: WinRateCombinationKey }) {
  const value = detail.returns[combination]
  const status = detail.statuses[combination]
  return (
    <span
      title={status === 'valid' ? undefined : status}
      className={value == null ? 'text-muted/50' : value > 0 ? 'text-emerald-400' : value < 0 ? 'text-danger' : 'text-muted'}
    >
      {percent(value)}
    </span>
  )
}

export function StrategyWinRatePanel({
  open,
  strategies,
  minDate,
  maxDate,
  defaultStart,
  defaultEnd,
  onClose,
}: Props) {
  const [selectedStrategies, setSelectedStrategies] = useState<string[]>([])
  const [startDate, setStartDate] = useState('')
  const [endDate, setEndDate] = useState('')
  const [boards, setBoards] = useState<string[]>(BOARD_OPTIONS)
  const [search, setSearch] = useState('')
  const [result, setResult] = useState<StrategyWinRateResult | null>(null)
  const [tab, setTab] = useState<'summary' | 'details'>('summary')
  const [page, setPage] = useState(1)

  useEffect(() => {
    if (!open) return
    setSelectedStrategies(strategies.map(strategy => strategy.id))
    const suggestedStart = defaultStart || dateBefore(defaultEnd || maxDate, 60)
    setStartDate(minDate && suggestedStart < minDate ? minDate : suggestedStart)
    setEndDate(defaultEnd || maxDate)
    setBoards([...BOARD_OPTIONS])
    setSearch('')
    setResult(null)
    setTab('summary')
    setPage(1)
  }, [open, strategies, minDate, defaultStart, defaultEnd, maxDate])

  const filteredStrategies = useMemo(() => {
    const query = search.trim().toLowerCase()
    if (!query) return strategies
    return strategies.filter(strategy =>
      strategy.name.toLowerCase().includes(query) || strategy.id.toLowerCase().includes(query),
    )
  }, [search, strategies])

  const selectedSet = useMemo(() => new Set(selectedStrategies), [selectedStrategies])
  const allVisibleSelected = filteredStrategies.length > 0 && filteredStrategies.every(strategy => selectedSet.has(strategy.id))
  const detailRows = result?.details ?? []
  const pageCount = Math.max(1, Math.ceil(detailRows.length / PAGE_SIZE))
  const visibleDetails = detailRows.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE)

  const calculate = useMutation({
    mutationFn: () => api.screenerWinRate({
      strategy_ids: selectedStrategies,
      start_date: startDate,
      end_date: endDate,
      boards,
    }),
    onSuccess: data => {
      setResult(data)
      setTab('summary')
      setPage(1)
    },
  })

  const toggleStrategy = (id: string) => {
    setSelectedStrategies(current => current.includes(id)
      ? current.filter(value => value !== id)
      : [...current, id])
  }

  const toggleBoard = (board: string) => {
    setBoards(current => current.includes(board)
      ? current.filter(value => value !== board)
      : [...current, board])
  }

  const toggleVisibleStrategies = () => {
    if (allVisibleSelected) {
      const visible = new Set(filteredStrategies.map(strategy => strategy.id))
      setSelectedStrategies(current => current.filter(id => !visible.has(id)))
    } else {
      setSelectedStrategies(current => Array.from(new Set([
        ...current,
        ...filteredStrategies.map(strategy => strategy.id),
      ])))
    }
  }

  const canCalculate = selectedStrategies.length > 0 && !!startDate && !!endDate && boards.length > 0 && startDate <= endDate

  return (
    <Modal
      onClose={onClose}
      labelledBy="strategy-win-rate-title"
      closeOnBackdrop={!calculate.isPending}
      panelClassName="w-[96vw] max-w-[1180px] max-h-[92vh] bg-surface border border-border rounded-card shadow-xl flex flex-col overflow-hidden"
    >
      <div className="flex items-center justify-between px-5 py-3 border-b border-border shrink-0">
        <div>
          <h2 id="strategy-win-rate-title" className="text-sm font-semibold text-foreground flex items-center gap-2">
            <BarChart3 className="h-4 w-4 text-accent" />
            策略胜率
          </h2>
          <p className="text-[11px] text-muted mt-1">T 日收盘选股，T+1 买入，T+2 卖出</p>
        </div>
        <button onClick={onClose} disabled={calculate.isPending} className="p-1.5 rounded-btn text-muted hover:text-foreground hover:bg-elevated disabled:opacity-40 cursor-pointer" title="关闭">
          <X className="h-4 w-4" />
        </button>
      </div>

      <div className="px-5 py-3 border-b border-border/70 grid grid-cols-1 lg:grid-cols-[1.35fr_0.8fr_1fr_auto] gap-3 shrink-0">
        <div className="min-w-0">
          <div className="flex items-center justify-between mb-1.5">
            <span className="text-[11px] font-medium text-secondary">策略</span>
            <span className="text-[10px] text-muted">已选 {selectedStrategies.length}/{strategies.length}</span>
          </div>
          <div className="flex items-center gap-1.5 mb-1.5">
            <div className="flex-1 relative">
              <Search className="absolute left-2 top-1/2 -translate-y-1/2 h-3 w-3 text-muted" />
              <input value={search} onChange={event => setSearch(event.target.value)} placeholder="搜索策略" className="w-full h-7 pl-7 pr-2 rounded-input border border-border bg-base text-xs text-foreground outline-none focus:border-accent/60" />
            </div>
            <button onClick={toggleVisibleStrategies} className="h-7 px-2 rounded-btn border border-border text-[10px] text-secondary hover:text-accent hover:border-accent/50 cursor-pointer whitespace-nowrap">
              {allVisibleSelected ? '清空当前' : '全选当前'}
            </button>
          </div>
          <div className="h-[84px] overflow-y-auto rounded-input border border-border bg-base/50 p-1 grid grid-cols-1 sm:grid-cols-2 gap-0.5">
            {filteredStrategies.map(strategy => {
              const checked = selectedSet.has(strategy.id)
              return (
                <label key={strategy.id} className="flex items-center gap-1.5 px-1.5 py-1 rounded-btn hover:bg-elevated cursor-pointer min-w-0">
                  <input type="checkbox" checked={checked} onChange={() => toggleStrategy(strategy.id)} className="accent-[var(--color-accent)] shrink-0" />
                  <span className="truncate text-[11px] text-secondary" title={strategy.name}>{strategy.name}</span>
                </label>
              )
            })}
            {filteredStrategies.length === 0 && <span className="col-span-2 text-[11px] text-muted text-center py-3">没有匹配策略</span>}
          </div>
        </div>

        <div>
          <span className="text-[11px] font-medium text-secondary block mb-1.5">信号日期</span>
          <div className="flex items-center gap-1.5">
            <DatePicker value={startDate} onChange={setStartDate} min={minDate} max={maxDate} buttonClassName="w-full justify-center" align="left" />
            <span className="text-muted text-xs">至</span>
            <DatePicker value={endDate} onChange={setEndDate} min={minDate} max={maxDate} buttonClassName="w-full justify-center" />
          </div>
          <span className="text-[10px] text-muted mt-1.5 block">可用数据：{minDate || '-'} 至 {maxDate || '-'}</span>
        </div>

        <div>
          <div className="flex items-center justify-between mb-1.5">
            <span className="text-[11px] font-medium text-secondary">市场</span>
            <button onClick={() => setBoards(boards.length === BOARD_OPTIONS.length ? [] : [...BOARD_OPTIONS])} className="text-[10px] text-accent hover:text-accent/80 cursor-pointer">
              {boards.length === BOARD_OPTIONS.length ? '清空' : '全选'}
            </button>
          </div>
          <div className="grid grid-cols-2 gap-1.5">
            {BOARD_OPTIONS.map(board => (
              <label key={board} className="flex items-center gap-1.5 text-[11px] text-secondary cursor-pointer">
                <input type="checkbox" checked={boards.includes(board)} onChange={() => toggleBoard(board)} className="accent-[var(--color-accent)]" />
                {board}
              </label>
            ))}
          </div>
        </div>

        <div className="flex items-end justify-end">
          <button
            onClick={() => calculate.mutate()}
            disabled={!canCalculate || calculate.isPending}
            className="inline-flex items-center justify-center gap-1.5 h-8 px-4 rounded-btn bg-accent text-white text-xs font-medium hover:bg-accent/90 disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer whitespace-nowrap"
          >
            {calculate.isPending ? <LoaderCircle className="h-3.5 w-3.5 animate-spin" /> : <BarChart3 className="h-3.5 w-3.5" />}
            {calculate.isPending ? '计算中' : '计算胜率'}
          </button>
        </div>
      </div>

      {calculate.isError && (
        <div className="mx-5 mt-3 px-3 py-2 rounded-btn border border-danger/30 bg-danger/10 text-danger text-xs shrink-0">
          {String((calculate.error as Error)?.message ?? '计算失败')}
        </div>
      )}

      <div className="flex-1 min-h-0 overflow-y-auto px-5 py-4">
        {!result && !calculate.isPending && !calculate.isError && (
          <div className="h-full min-h-[180px] flex flex-col items-center justify-center text-muted gap-2">
            <BarChart3 className="h-8 w-8 text-accent/30" />
            <span className="text-xs">选择策略、日期和市场后开始计算</span>
          </div>
        )}

        {calculate.isPending && (
          <div className="h-full min-h-[180px] flex flex-col items-center justify-center text-muted gap-3">
            <LoaderCircle className="h-7 w-7 text-accent animate-spin" />
            <span className="text-xs">正在按历史交易日运行策略并匹配价格</span>
          </div>
        )}

        {result && !calculate.isPending && (
          <div className="space-y-4">
            <div className="flex items-center justify-between gap-3 flex-wrap">
              <div className="text-xs text-secondary">
                共 <span className="text-accent font-semibold num">{result.summary.signal_count}</span> 条策略信号
                <span className="text-muted ml-2">{result.config.start_date} 至 {result.config.end_date}</span>
              </div>
              <div className="inline-flex h-7 rounded-btn border border-border bg-base overflow-hidden">
                {(['summary', 'details'] as const).map(value => (
                  <button key={value} onClick={() => setTab(value)} className={`px-3 text-xs cursor-pointer ${tab === value ? 'bg-accent/15 text-accent' : 'text-muted hover:text-secondary'}`}>
                    {value === 'summary' ? '汇总' : `明细 ${result.details.length}`}
                  </button>
                ))}
              </div>
            </div>

            {tab === 'summary' ? (
              <>
                <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-2">
                  {COMBINATION_KEYS.map(key => {
                    const stats = result.summary.combinations[key]
                    return (
                      <div key={key} className="border border-border rounded-btn bg-base/50 p-3">
                        <div className="text-[11px] text-secondary truncate">{result.combination_labels[key]}</div>
                        <div className="mt-2 flex items-baseline gap-2">
                          <span className="text-xl font-semibold num text-accent">{percent(stats.win_rate)}</span>
                          <span className="text-[10px] text-muted">胜率</span>
                        </div>
                        <div className="mt-1 text-[10px] text-muted num">有效 {stats.valid} · 胜 {stats.wins} · 负 {stats.losses} · 平 {stats.flats}</div>
                         <div className="mt-1 text-[10px] text-muted num">平均 {percent(stats.avg_return)} · 跳过 {stats.skipped}</div>
                      </div>
                    )
                  })}
                </div>

                {COMBINATION_KEYS.some(key => Object.keys(result.summary.combinations[key].skip_reasons ?? {}).length > 0) && (
                  <div className="text-[10px] text-muted leading-5">
                    跳过原因：{COMBINATION_KEYS.flatMap(key => {
                      const reason = skipReasons(result.summary.combinations[key])
                      return reason ? [`${result.combination_labels[key]}：${reason}`] : []
                    }).join(' · ')}
                  </div>
                )}

                <div className="overflow-x-auto border border-border rounded-btn">
                  <table className="w-full min-w-[850px] text-xs">
                    <thead className="bg-base/80 text-[10px] text-muted">
                      <tr>
                        <th className="text-left px-3 py-2 font-medium">策略</th>
                        <th className="text-right px-3 py-2 font-medium">信号数</th>
                        {COMBINATION_KEYS.map(key => <th key={key} className="text-right px-3 py-2 font-medium">{result.combination_labels[key]}</th>)}
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-border/60">
                      {result.summary.strategies.map(strategy => (
                        <tr key={strategy.strategy_id} className="hover:bg-elevated/40">
                          <td className="px-3 py-2 text-secondary">{strategy.strategy_name}<span className="ml-1 text-[10px] text-muted font-mono">{strategy.strategy_id}</span></td>
                          <td className="px-3 py-2 text-right num text-secondary">{strategy.signal_count}</td>
                          {COMBINATION_KEYS.map(key => {
                            const stats = strategy.combinations[key]
                            return <td key={key} className="px-3 py-2 text-right num"><span className={stats.win_rate == null ? 'text-muted' : 'text-accent'}>{percent(stats.win_rate)}</span><span className="text-[10px] text-muted ml-1">({stats.valid})</span></td>
                          })}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </>
            ) : (
              <>
                <div className="overflow-x-auto border border-border rounded-btn">
                  <table className="w-full min-w-[1120px] text-[11px]">
                    <thead className="bg-base/80 text-[10px] text-muted">
                      <tr>
                        <th className="text-left px-2.5 py-2 font-medium">信号日</th>
                        <th className="text-left px-2.5 py-2 font-medium">策略</th>
                        <th className="text-left px-2.5 py-2 font-medium">股票</th>
                        <th className="text-left px-2.5 py-2 font-medium">板块</th>
                        <th className="text-left px-2.5 py-2 font-medium">买入日</th>
                        <th className="text-left px-2.5 py-2 font-medium">卖出日</th>
                        <th className="text-right px-2.5 py-2 font-medium">开买开卖</th>
                        <th className="text-right px-2.5 py-2 font-medium">开买收卖</th>
                        <th className="text-right px-2.5 py-2 font-medium">收买开卖</th>
                        <th className="text-right px-2.5 py-2 font-medium">收买收卖</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-border/60">
                      {visibleDetails.map(detail => (
                        <tr key={`${detail.strategy_id}-${detail.symbol}-${detail.signal_date}`} className="hover:bg-elevated/40">
                          <td className="px-2.5 py-2 num text-secondary">{detail.signal_date}</td>
                          <td className="px-2.5 py-2 text-secondary max-w-[150px] truncate" title={detail.strategy_name}>{detail.strategy_name}</td>
                          <td className="px-2.5 py-2 text-foreground whitespace-nowrap">{detail.name || '-'} <span className="text-muted font-mono">{detail.symbol}</span></td>
                          <td className="px-2.5 py-2 text-muted">{detail.board}</td>
                           <td className="px-2.5 py-2 num text-muted">{detail.buy_date ?? '-'}</td>
                           <td className="px-2.5 py-2 num text-muted">{detail.sell_date ?? '-'}</td>
                          {COMBINATION_KEYS.map(key => <td key={key} className="px-2.5 py-2 text-right num"><ReturnCell detail={detail} combination={key} /></td>)}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  {visibleDetails.length === 0 && <div className="py-10 text-center text-xs text-muted">没有可展示的明细</div>}
                </div>
                <div className="flex items-center justify-end gap-2 text-[11px] text-muted">
                  <span>第 {page} / {pageCount} 页</span>
                  <button disabled={page <= 1} onClick={() => setPage(value => Math.max(1, value - 1))} className="p-1 rounded-btn border border-border hover:text-accent disabled:opacity-30 cursor-pointer disabled:cursor-not-allowed"><ChevronLeft className="h-3.5 w-3.5" /></button>
                  <button disabled={page >= pageCount} onClick={() => setPage(value => Math.min(pageCount, value + 1))} className="p-1 rounded-btn border border-border hover:text-accent disabled:opacity-30 cursor-pointer disabled:cursor-not-allowed"><ChevronRight className="h-3.5 w-3.5" /></button>
                </div>
              </>
            )}
          </div>
        )}
      </div>
    </Modal>
  )
}
